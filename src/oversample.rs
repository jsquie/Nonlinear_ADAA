use core::panic;

// use crate::oversample_stage::OversampleStage;
use crate::oversample::os_filter_constants::FILTER_TAPS;
use crate::oversample::oversample_stage::OversampleStage;
use ndarray::Array1;
use nih_plug::prelude::*;
use num_traits::Float;

mod os_filter_constants;
mod oversample_stage;
// use crate::oversample_stage::OversampleStage;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct OSFactorScale {
    factor: u32,
    scale: u32,
}

#[allow(dead_code)]
impl OSFactorScale {
    pub const TWO_TIMES: OSFactorScale = OSFactorScale {
        factor: 1,
        scale: 2,
    };
    pub const FOUR_TIMES: OSFactorScale = OSFactorScale {
        factor: 2,
        scale: 2_u32.pow(2),
    };
    pub const EIGHT_TIMES: OSFactorScale = OSFactorScale {
        factor: 3,
        scale: 2_u32.pow(3),
    };
    pub const SIXTEEN_TIMES: OSFactorScale = OSFactorScale {
        factor: 4,
        scale: 2_u32.pow(4),
    };
}

#[derive(Enum, Debug, Copy, Clone, PartialEq)]
pub enum OversampleFactor {
    #[id = "2x"]
    #[name = "Two Times"]
    TwoTimes,
    #[id = "4x"]
    #[name = "Four Times"]
    FourTimes,
    #[id = "8x"]
    #[name = "Eight Times"]
    EightTimes,
    #[id = "16x"]
    #[name = "Sixteen Times"]
    SixteenTimes,
}

#[allow(dead_code)]
enum SampleRole {
    UpSample,
    DownSample,
}

#[allow(dead_code)]
pub struct Oversample<T>
where
    T: Float + Copy + From<f32> + 'static,
{
    buff_size: u32,
    factor: OversampleFactor,
    factor_scale: OSFactorScale,
    up_stages: Vec<OversampleStage<T>>,
    down_stages: Vec<OversampleStage<T>>,
    kernel: Array1<T>,
}

#[allow(dead_code)]
impl<T> Oversample<T>
where
    T: Float + Copy + From<f32> + 'static,
{
    pub fn new(initial_factor: OversampleFactor, initial_buff_size: u32) -> Self {
        let new_factor = match initial_factor {
            OversampleFactor::TwoTimes => OSFactorScale::TWO_TIMES,
            OversampleFactor::FourTimes => OSFactorScale::FOUR_TIMES,
            OversampleFactor::EightTimes => OSFactorScale::EIGHT_TIMES,
            OversampleFactor::SixteenTimes => OSFactorScale::SIXTEEN_TIMES,
        };

        Oversample {
            factor: initial_factor,
            factor_scale: new_factor,
            buff_size: initial_buff_size,
            up_stages: (1..=new_factor.factor)
                .map(|factor| {
                    OversampleStage::new(
                        (initial_buff_size * 2_u32.pow(factor)) as usize,
                        SampleRole::UpSample,
                    )
                })
                .collect::<Vec<_>>(),
            down_stages: (1..=new_factor.factor)
                .map(|factor| {
                    OversampleStage::new(
                        (initial_buff_size * 2_u32.pow(new_factor.factor - factor)) as usize,
                        SampleRole::DownSample,
                    )
                })
                .collect::<Vec<_>>(),
            kernel: Array1::from_iter(FILTER_TAPS.into_iter().map(|x| (x as f32).into())),
        }
    }

    pub fn reset(&mut self) {
        for (up_stage, down_stage) in self.up_stages.iter_mut().zip(self.down_stages.iter_mut()) {
            up_stage.reset();
            down_stage.reset();
        }
    }

    pub fn process_up(&mut self, input: &[T]) -> &[T] {
        let mut stages = self.up_stages.iter_mut();

        let mut last_stage = match stages.next() {
            Some(stage) => {
                stage.process_up(input, &self.kernel);
                &stage.data
            }
            None => panic!(
                "There must be at least one up sample stage for processing the raw input data"
            ),
        };

        for up_stage in stages {
            up_stage.process_up(&last_stage, &self.kernel);
            last_stage = &up_stage.data;
        }
        last_stage
    }

    pub fn process_down(&mut self, input: &[T]) -> &[T] {
        let mut stages = self.down_stages.iter_mut();

        let mut last_stage = match stages.next() {
            Some(stage) => {
                stage.process_down(input, &self.kernel);
                &stage.data
            }
            None => panic!(
                "There must be at least one up sample stage for processing the raw input data"
            ),
        };

        for down_stage in stages {
            down_stage.process_up(last_stage, &self.kernel);
            last_stage = &down_stage.data;
        }

        last_stage
    }
}

#[cfg(test)]
mod tests {

    use crate::oversample::*;

    #[test]
    fn test_create_os_2x() {
        let os = Oversample::<f32>::new(OversampleFactor::TwoTimes, 4);
        assert_eq!(os.up_stages.len(), 1);
        assert_eq!(os.down_stages.len(), 1);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.down_stages[0].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        assert_eq!(os_64.up_stages.len(), 1);
        assert_eq!(os_64.down_stages.len(), 1);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.down_stages[0].data.len(), 4);
    }

    #[test]
    fn test_create_os_4x() {
        let os = Oversample::<f32>::new(OversampleFactor::FourTimes, 4);
        assert_eq!(os.up_stages.len(), 2);
        assert_eq!(os.down_stages.len(), 2);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.down_stages[0].data.len(), 8);
        assert_eq!(os.down_stages[1].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        assert_eq!(os_64.up_stages.len(), 2);
        assert_eq!(os_64.down_stages.len(), 2);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.down_stages[0].data.len(), 8);
        assert_eq!(os_64.down_stages[1].data.len(), 4);
    }

    #[test]
    fn test_create_os_8x() {
        let os = Oversample::<f32>::new(OversampleFactor::EightTimes, 4);
        assert_eq!(os.up_stages.len(), 3);
        assert_eq!(os.down_stages.len(), 3);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.up_stages[2].data.len(), 32);
        assert_eq!(os.down_stages[0].data.len(), 16);
        assert_eq!(os.down_stages[1].data.len(), 8);
        assert_eq!(os.down_stages[2].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        assert_eq!(os_64.up_stages.len(), 3);
        assert_eq!(os_64.down_stages.len(), 3);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.up_stages[2].data.len(), 32);
        assert_eq!(os_64.down_stages[0].data.len(), 16);
        assert_eq!(os_64.down_stages[1].data.len(), 8);
        assert_eq!(os_64.down_stages[2].data.len(), 4);
    }

    #[test]
    fn test_create_os_16x() {
        let os = Oversample::<f32>::new(OversampleFactor::SixteenTimes, 4);
        assert_eq!(os.up_stages.len(), 4);
        assert_eq!(os.down_stages.len(), 4);
        assert_eq!(os.up_stages[0].data.len(), 8);
        assert_eq!(os.up_stages[1].data.len(), 16);
        assert_eq!(os.up_stages[2].data.len(), 32);
        assert_eq!(os.up_stages[3].data.len(), 64);
        assert_eq!(os.down_stages[0].data.len(), 32);
        assert_eq!(os.down_stages[1].data.len(), 16);
        assert_eq!(os.down_stages[2].data.len(), 8);
        assert_eq!(os.down_stages[3].data.len(), 4);
        let os_64 = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        assert_eq!(os_64.up_stages.len(), 4);
        assert_eq!(os_64.down_stages.len(), 4);
        assert_eq!(os_64.up_stages[0].data.len(), 8);
        assert_eq!(os_64.up_stages[1].data.len(), 16);
        assert_eq!(os_64.up_stages[2].data.len(), 32);
        assert_eq!(os_64.up_stages[3].data.len(), 64);
        assert_eq!(os_64.down_stages[0].data.len(), 32);
        assert_eq!(os_64.down_stages[1].data.len(), 16);
        assert_eq!(os_64.down_stages[2].data.len(), 8);
        assert_eq!(os_64.down_stages[3].data.len(), 4);
    }

    #[test]
    fn test_small_up_sample_2x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result = os.process_up(sig);
        let expected_result: &[f64] = &[
            -0.012943094819578109,
            0.0,
            0.013577449569054703,
            0.0,
            -0.014268251144141814,
            0.0,
            0.015023742543533445,
            0.0,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_4x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result = os.process_up(sig);

        let expected_result: &[f64] = &[
            0.00016752370350858967,
            0.0,
            -0.00017573421718031495,
            0.0,
            8.941110287866394e-06,
            0.0,
            -1.0106587485659248e-05,
            0.0,
            0.00019614686008995804,
            0.0,
            -0.00020680688566223008,
            0.0,
            2.411980294562719e-05,
            0.0,
            -2.7654982156956038e-05,
            0.0,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_8x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result = os.process_up(sig);
        let expected_result: &[f64] = &[
            -2.168275179038566e-06,
            0.0,
            2.2745446360091485e-06,
            0.0,
            -1.1572563824815977e-07,
            0.0,
            1.3081052012924913e-07,
            0.0,
            -2.6420277269769886e-07,
            0.0,
            2.9069865914485977e-07,
            0.0,
            -1.9078742252992996e-07,
            0.0,
            2.207193743894453e-07,
            0.0,
            -2.7956397603268105e-06,
            0.0,
            2.9642240835066686e-06,
            0.0,
            -4.775282888187613e-07,
            0.0,
            5.627502326401241e-07,
            0.0,
            -9.8157074847025e-07,
            0.0,
            1.1325599486140385e-06,
            0.0,
            -9.674399702727956e-07,
            0.0,
            1.2024669215155766e-06,
            0.0,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn test_small_up_sample_16x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        let sig: &[f64] = &[1., 0., 0., 0.];

        let result = os.process_up(sig);

        let expected_result: &[f64] = &[
            -2.168275179038566e-06,
            0.0,
            2.2745446360091485e-06,
            0.0,
            -1.1572563824815977e-07,
            0.0,
            1.3081052012924913e-07,
            0.0,
            -2.6420277269769886e-07,
            0.0,
            2.9069865914485977e-07,
            0.0,
            -1.9078742252992996e-07,
            0.0,
            2.207193743894453e-07,
            0.0,
            -2.7956397603268105e-06,
            0.0,
            2.9642240835066686e-06,
            0.0,
            -4.775282888187613e-07,
            0.0,
            5.627502326401241e-07,
            0.0,
            -9.8157074847025e-07,
            0.0,
            1.1325599486140385e-06,
            0.0,
            -9.674399702727956e-07,
            0.0,
            1.2024669215155766e-06,
            0.0,
            -4.593246554270411e-06,
            0.0,
            5.202622639996159e-06,
            0.0,
            -2.7888950942274918e-06,
            0.0,
            3.909224522374513e-06,
            0.0,
            -6.555036081593156e-06,
            0.0,
            1.0622833522474865e-05,
            0.0,
            -2.0835205101646454e-05,
            0.0,
            8.627421683977256e-05,
            0.00016858237728001698,
            0.00013899531362654408,
            0.0,
            -0.0001415748393474062,
            -0.0001768447776716043,
            -9.52287845218878e-05,
            0.0,
            3.1070390543988444e-05,
            8.997614046743225e-06,
            -1.092756459531192e-05,
            0.0,
            1.0974384173624137e-05,
            -1.0170456531445681e-05,
            -3.3768688848351426e-05,
            0.0,
            0.00010526245511897318,
            0.00019738641921965706,
        ];

        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_2x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::TwoTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result = os.process_up(sig);
        let expected_result: &[f64] = &[
            -0.0064715474097890545,
            0.006788724784527351,
            -0.007134125572070907,
            0.007511871271766723,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_4x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::FourTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result = os.process_up(sig);
        let expected_result: &[f64] = &[
            4.188092587714742e-05,
            2.2352775719665985e-06,
            4.903671502248951e-05,
            6.029950736406798e-06,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_8x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::EightTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result = os.process_up(sig);

        let expected_result: &[f64] = &[
            -2.7103439737982077e-07,
            -3.302534658721244e-08,
            -3.494549700408512e-07,
            -1.2269634355878129e-07,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }

    #[test]
    fn down_sample_16x() {
        let mut os = Oversample::<f64>::new(OversampleFactor::SixteenTimes, 4);
        let sig_vec = &vec![vec![1.], vec![0.; 7]]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let sig: &[f64] = sig_vec.as_slice();

        let result = os.process_up(sig);
        let expected_result: &[f64] = &[
            1.7540119523271163e-09,
            4.2153647525395777e-10,
            3.276916414167012e-09,
            -1.1588005093991977e-07,
        ];
        for (r, e) in result.iter().zip(expected_result.iter()) {
            assert!(
                (r - e).abs() < 1e-7,
                "Assertion failed: res: {}, expected: {}",
                r,
                e
            )
        }
    }
}
