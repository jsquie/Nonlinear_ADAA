use nih_plug::prelude::*;
use Nonlinear_ADAA::NonlinearAdaa;
fn main() {
    nih_export_standalone::<NonlinearAdaa>();
}
