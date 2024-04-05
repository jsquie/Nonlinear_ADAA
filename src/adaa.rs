use nih_plug::prelude::*;

#[derive(Enum, Debug, Copy, PartialEq)]
pub enum NLProc {
    #[id = "hard clip"]
    #[name = "Hard Clip"]
    HardClip,
    #[id = "tanh"]
    #[name = "Tanh"]
    Tanh,
}

impl Clone for NLProc {
    fn clone_from(&mut self, source: &Self) {
        match source {
            NLProc::Tanh => *self = NLProc::Tanh,
            NLProc::HardClip => *self = NLProc::HardClip,
        }
    }

    fn clone(&self) -> Self {
        match self {
            NLProc::HardClip => NLProc::HardClip,
            NLProc::Tanh => NLProc::Tanh,
        }
    }
}

#[inline]
fn nl_func(proc: &NLProc, val: f64) -> f64 {
    match proc {
        NLProc::HardClip => {
            if val.abs() > 1.0 {
                val.signum()
            } else {
                val
            }
        }
        NLProc::Tanh => val.tanh(),
    }
}

#[inline]
fn nl_func_f1(proc: &NLProc, val: f64) -> f64 {
    match proc {
        NLProc::HardClip => {
            if val.abs() <= 1.0 {
                0.5 * val.powi(2)
            } else {
                (val * val.signum()) - 0.5
            }
        }
        NLProc::Tanh => f64::ln(f64::cosh(val)),
    }
}

#[inline]
fn nl_func_f2(proc: &NLProc, val: f64) -> f64 {
    match proc {
        NLProc::HardClip => {
            if val.abs() <= 1.0 {
                val.powi(3) * (1. / 6.)
            } else {
                (((val.powi(2) * 0.5) + (1. / 6.)) * val.signum()) - (val * 0.5)
            }
        }
        NLProc::Tanh => {
            let exp: f64 = (-2. * val).exp();
            let poly: f64 = polylog::Li2::li2(&(exp * -1.));
            let prt2: f64 = val * (val + 2.0 * (exp + 1.).ln());
            let prt3: f64 = 2.0 * val.cosh().ln();
            0.5 * (poly - prt2 - prt3) + (std::f64::consts::PI.powi(2) / 24.0)
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ADAAFirst {
    x1: f64,
    ad1_x1: f64,
    proc: NLProc,
}

impl ADAAFirst {
    pub fn new(pr: NLProc) -> ADAAFirst {
        ADAAFirst {
            x1: 0.0,
            ad1_x1: 0.0,
            proc: pr,
        }
    }

    pub fn reset(&self, pr: NLProc) -> Self {
        Self {
            x1: 0.0,
            ad1_x1: 0.0,
            proc: pr,
        }
    }

    pub fn next_adaa(&mut self, val: &f32) -> f32 {
        let s: f64 = *val as f64;
        let diff: f64 = s - self.x1;
        let ill_condition: bool = diff.abs() < 1e-5;
        let ad1_x0: f64 = nl_func_f1(&self.proc, s);
        let result: f64;

        if ill_condition {
            result = nl_func(&self.proc, (s + self.x1) / 2.);
        } else {
            result = (ad1_x0 - self.ad1_x1) / diff;
        }

        self.x1 = s;
        self.ad1_x1 = ad1_x0;

        result as f32
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ADAASecond {
    x1: f64,
    x2: f64,
    ad2_x0: f64,
    ad2_x1: f64,
    d2: f64,
    proc: NLProc,
}

impl ADAASecond {
    pub fn new(pr: NLProc) -> ADAASecond {
        ADAASecond {
            x1: 0.0,
            x2: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            d2: 0.0,
            proc: pr,
        }
    }

    pub fn reset(&self, pr: NLProc) -> Self {
        Self {
            x1: 0.0,
            x2: 0.0,
            ad2_x0: 0.0,
            ad2_x1: 0.0,
            d2: 0.0,
            proc: pr,
        }
    }

    #[inline]
    fn calc_d(&mut self, val: &f64) -> f64 {
        self.ad2_x0 = nl_func_f2(&self.proc, *val);

        let ill_condition: bool = f64::abs(val - self.x1) < 1e-5;

        if ill_condition {
            nl_func_f1(&self.proc, 0.5 * (val + self.x1))
        } else {
            (self.ad2_x0 - self.ad2_x1) / (val - self.x1)
        }
    }

    #[inline]
    fn fallback(&self, val: &f64) -> f64 {
        let xbar: f64 = 0.5 * (*val + self.x2);
        let delta: f64 = xbar - self.x1;
        let ill_condition: bool = delta.abs() < 1e-5;

        if ill_condition {
            nl_func(&self.proc, 0.5 * (xbar + self.x1))
        } else {
            (2.0 / delta)
                * (nl_func_f1(&self.proc, xbar)
                    + (self.ad2_x1 - nl_func_f2(&self.proc, xbar)) / delta)
        }
    }

    pub fn next_adaa(&mut self, val: &f32) -> f32 {
        let s: f64 = *val as f64;
        let res: f64;
        let d1: f64 = self.calc_d(&s);
        let ill_condition: bool = f64::abs(s - self.x2) < 1e-5;

        if ill_condition {
            res = self.fallback(&s);
        } else {
            res = (2.0 / (s - self.x2)) * (d1 - self.d2)
        }

        self.d2 = d1;
        self.x2 = self.x1;
        self.x1 = s;
        self.ad2_x1 = self.ad2_x0;

        res as f32
    }
}
