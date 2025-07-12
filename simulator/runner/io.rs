use std::{
    cell::{Cell, RefCell},
    sync::Arc,
};

use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use turso_core::{Clock, Instant, OpenFlags, PlatformIO, Result, IO};

use crate::{model::FAULT_ERROR_MSG, runner::file::SimulatorFile};

pub(crate) struct SimulatorIO {
    pub(crate) inner: Box<dyn IO>,
    pub(crate) fault: Cell<bool>,
    pub(crate) files: RefCell<Vec<Arc<SimulatorFile>>>,
    pub(crate) rng: RefCell<ChaCha8Rng>,
    pub(crate) nr_run_once_faults: Cell<usize>,
    pub(crate) page_size: usize,
    seed: u64,
    latency_probability: usize,
    
    pub(crate) io_callback_drop_mode: Cell<bool>,
    pub(crate) operation_count: Cell<usize>,
    pub(crate) fail_after_operations: Cell<usize>,
    pub(crate) pending_operations: RefCell<Vec<String>>,
}

unsafe impl Send for SimulatorIO {}
unsafe impl Sync for SimulatorIO {}

impl SimulatorIO {
    pub(crate) fn new(seed: u64, page_size: usize, latency_probability: usize) -> Result<Self> {
        let inner = Box::new(PlatformIO::new()?);
        let fault = Cell::new(false);
        let files = RefCell::new(Vec::new());
        let rng = RefCell::new(ChaCha8Rng::seed_from_u64(seed));
        let nr_run_once_faults = Cell::new(0);
        Ok(Self {
            inner,
            fault,
            files,
            rng,
            nr_run_once_faults,
            page_size,
            seed,
            latency_probability,
            io_callback_drop_mode: Cell::new(false),
            operation_count: Cell::new(0),
            fail_after_operations: Cell::new(2), // Default: fail after 2 operations
            pending_operations: RefCell::new(Vec::new()),
        })
    }

    pub(crate) fn inject_fault(&self, fault: bool) {
        self.fault.replace(fault);
        for file in self.files.borrow().iter() {
            file.inject_fault(fault);
        }
    }

    /// Enable IO callback drop bug reproduction mode
    pub(crate) fn enable_io_callback_drop_mode(&self, fail_after: usize) {
        self.io_callback_drop_mode.set(true);
        self.fail_after_operations.set(fail_after);
        self.operation_count.set(0);
        self.pending_operations.borrow_mut().clear();
        tracing::info!("Enabled IO callback drop mode: will fail after {} operations", fail_after);
    }


    pub(crate) fn disable_io_callback_drop_mode(&self) {
        self.io_callback_drop_mode.set(false);
        tracing::info!("Disabled IO callback drop mode");
    }


    pub(crate) fn should_inject_io_callback_drop_fault(&self, operation: &str) -> bool {
        if !self.io_callback_drop_mode.get() {
            return false;
        }

        let count = self.operation_count.get();
        self.operation_count.set(count + 1);
        
        self.pending_operations.borrow_mut().push(format!("{}_{}", operation, count));
        
        let should_fail = count >= self.fail_after_operations.get();
        
        if should_fail {
            tracing::warn!(
                "Injecting fault for run_once bug reproduction: operation {} ({}), count: {}, pending: {:?}",
                operation, count, count, self.pending_operations.borrow()
            );
        } else {
            tracing::debug!(
                "Operation {} ({}) proceeding normally, count: {}, pending: {:?}",
                operation, count, count, self.pending_operations.borrow()
            );
        }
        
        should_fail
    }

    pub(crate) fn print_stats(&self) {
        tracing::info!("run_once faults: {}", self.nr_run_once_faults.get());
        if self.io_callback_drop_mode.get() {
            tracing::info!("IO callback drop mode: enabled, operation_count: {}, fail_after: {}", 
                          self.operation_count.get(), self.fail_after_operations.get());
            tracing::info!("pending operations: {:?}", self.pending_operations.borrow());
        }
        for file in self.files.borrow().iter() {
            tracing::info!("\n===========================\n{}", file.stats_table());
        }
    }
}

impl Clock for SimulatorIO {
    fn now(&self) -> Instant {
        Instant {
            secs: 1704067200, // 2024-01-01 00:00:00 UTC
            micros: 0,
        }
    }
}

impl IO for SimulatorIO {
    fn open_file(
        &self,
        path: &str,
        flags: OpenFlags,
        _direct: bool,
    ) -> Result<Arc<dyn turso_core::File>> {
        let inner = self.inner.open_file(path, flags, false)?;
        let file = Arc::new(SimulatorFile {
            inner,
            fault: Cell::new(false),
            nr_pread_faults: Cell::new(0),
            nr_pwrite_faults: Cell::new(0),
            nr_sync_faults: Cell::new(0),
            nr_pread_calls: Cell::new(0),
            nr_pwrite_calls: Cell::new(0),
            nr_sync_calls: Cell::new(0),
            page_size: self.page_size,
            rng: RefCell::new(ChaCha8Rng::seed_from_u64(self.seed)),
            latency_probability: self.latency_probability,
            sync_completion: RefCell::new(None),
            io_ref: self as *const SimulatorIO,
        });
        self.files.borrow_mut().push(file.clone());
        Ok(file)
    }

    fn wait_for_completion(&self, c: Arc<turso_core::Completion>) -> Result<()> {
        while !c.is_completed() {
            self.run_once()?;
        }
        Ok(())
    }

    fn run_once(&self) -> Result<()> {
        if self.fault.get() {
            self.nr_run_once_faults
                .replace(self.nr_run_once_faults.get() + 1);
            return Err(turso_core::LimboError::InternalError(
                FAULT_ERROR_MSG.into(),
            ));
        }
        self.inner.run_once()?;
        Ok(())
    }

    fn generate_random_number(&self) -> i64 {
        self.rng.borrow_mut().next_u64() as i64
    }

    fn get_memory_io(&self) -> Arc<turso_core::MemoryIO> {
        todo!()
    }
}
