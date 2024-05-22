use core::fmt;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::Deserialize;

use jdsp::{AntiderivativeOrder, NonlinearProcessor, ProcStateTransition, ProcessorStyle};
use jdsp::{Oversample, OversampleFactor};

const ERR_TOL: f32 = 1e-5;

#[allow(dead_code)]
struct TotalResult {
    incorrect_results: Vec<Option<ADAAResult>>,
    num_incorrect_results: Option<usize>,
    num_total_tests: usize,
    perc_correct_results: Option<f32>,
    avg_difference: Option<f32>,
}

impl TotalResult {
    pub fn new(results: Vec<Option<ADAAResult>>) -> Self {
        TotalResult {
            incorrect_results: results
                .clone()
                .into_iter()
                .filter(|v| v.is_some())
                .collect(),
            num_incorrect_results: None,
            num_total_tests: results.len(),
            perc_correct_results: None,
            avg_difference: None,
        }
    }

    pub fn initialize(&mut self) {
        self.num_incorrect_results = if self.incorrect_results.len() > 0 {
            self.avg_difference = Some(
                self.incorrect_results
                    .clone()
                    .into_iter()
                    .fold(0.0, |acc, v| acc + v.unwrap().difference)
                    / self.incorrect_results.len() as f32,
            );
            Some(self.incorrect_results.len())
        } else {
            None
        };
        self.perc_correct_results = Some(
            1. - (self.num_incorrect_results.unwrap_or(0) as f32 / self.num_total_tests as f32),
        );
    }
}

impl fmt::Debug for TotalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TotalResult: \n\tnum_incorrect: {}\n\tperc_correct: {}%\n\tavg difference: {}\n\tmax difference: {}\n\tincorrect: {:?}",
            self.num_incorrect_results.unwrap_or(0),
            self.perc_correct_results.unwrap_or(100.0),
            self.avg_difference.unwrap_or(0.0),
            self.incorrect_results.clone().into_iter().fold(0.0, |acc, v| v.unwrap().difference.max(acc)),
            self.incorrect_results
                .clone()
                .into_iter()
                .map(|v| format!("{:?}",
                    v.unwrap()))
        )
    }
}

#[derive(Clone)]
#[allow(dead_code)]
struct ADAAResult {
    my_result: f32,
    expected_result: f32,
    difference: f32,
}

impl fmt::Debug for ADAAResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "My result: {}, expected_result: {}, difference: {}\n",
            self.my_result, self.expected_result, self.difference
        )
    }
}

fn check_results(result: &[f32], expected_result: &[f32]) {
    let results: Vec<Option<ADAAResult>> = result
        .into_iter()
        .zip(expected_result.iter())
        .map(|(r, e)| {
            let res = (r - e).abs() < ERR_TOL;
            if res {
                None
            } else {
                Some(ADAAResult {
                    my_result: *r,
                    expected_result: *e,
                    difference: (r - e).abs(),
                })
            }
        })
        .collect();

    let mut total = TotalResult::new(results.clone());
    total.initialize();

    assert!(total.num_incorrect_results.unwrap_or(0) == 0, "{:?}", total);
}

#[derive(Deserialize, Debug)]
struct TestCase {
    name: String,
    input: Vec<f32>,
    expected_output: Vec<f32>,
}

fn read_test_case_from_file<P: AsRef<Path>>(path: P) -> Result<TestCase, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let u = serde_json::from_reader(reader)?;

    Ok(u)
}

#[test]
fn test_2x_tanh_ad1() {
    let mut os = Oversample::new(OversampleFactor::TwoTimes, 32);
    os.initialize_oversample_stages();
    let mut ad = NonlinearProcessor::new();
    ad.change_state(ProcStateTransition::ChangeStyle(ProcessorStyle::Tanh));

    let mut test_case = read_test_case_from_file("./tests/json_test_data/tanh_2x_ad1.json")
        .expect("File path incorrect");
    let mut output = vec![0.0_f32; 64];
    let mut result = vec![0.0_f32; 32];

    os.process_up(&mut test_case.input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.process(*v * 10.0));

    os.process_down(&mut output, &mut result);

    check_results(&result, &test_case.expected_output);
}

fn run_test_case(
    ad: &mut NonlinearProcessor,
    factor: OversampleFactor,
    size: usize,
    file_name: &str,
) {
    let mut os = Oversample::new(factor, size);
    os.initialize_oversample_stages();

    let mut test_case = read_test_case_from_file(file_name).expect("File not found");

    let mut output = match factor {
        OversampleFactor::TwoTimes => vec![0.0_f32; size * (2_u32.pow(1) as usize)],
        OversampleFactor::FourTimes => vec![0.0_f32; size * (2_u32.pow(2) as usize)],
        OversampleFactor::EightTimes => vec![0.0_f32; size * (2_u32.pow(3) as usize)],
        OversampleFactor::SixteenTimes => vec![0.0_f32; size * (2_u32.pow(4) as usize)],
    };
    let mut result = vec![0.0_f32; size];

    os.process_up(&mut test_case.input, &mut output);

    output.iter_mut().for_each(|v| *v = ad.process(*v * 10.0));

    os.process_down(&mut output, &mut result);

    check_results(&result, &test_case.expected_output);
}

#[test]
fn test_2x_tanh_ad2() {
    let mut proc = NonlinearProcessor::new();
    proc.change_state(ProcStateTransition::ChangeStyle(ProcessorStyle::Tanh));
    proc.change_state(ProcStateTransition::ChangeOrder(
        AntiderivativeOrder::SecondOrder,
    ));
    run_test_case(
        &mut proc,
        OversampleFactor::TwoTimes,
        32,
        "./tests/json_test_data/tanh_2x_ad2.json",
    );
}

#[test]
fn test_4x_tanh_ad1() {
    let mut proc = NonlinearProcessor::new();
    proc.change_state(ProcStateTransition::ChangeStyle(ProcessorStyle::Tanh));
    run_test_case(
        &mut proc,
        OversampleFactor::FourTimes,
        64,
        "./tests/json_test_data/tanh_4x_ad1.json",
    );
}

#[test]
fn test_4x_tanh_ad2() {
    let mut proc = NonlinearProcessor::new();
    proc.change_state(ProcStateTransition::ChangeStyle(ProcessorStyle::Tanh));
    proc.change_state(ProcStateTransition::ChangeOrder(
        AntiderivativeOrder::SecondOrder,
    ));
    run_test_case(
        &mut proc,
        OversampleFactor::FourTimes,
        64,
        "./tests/json_test_data/tanh_4x_ad2.json",
    );
}

#[test]
fn test_2x_hc_ad1() {
    let mut proc = NonlinearProcessor::new();
    run_test_case(
        &mut proc,
        OversampleFactor::TwoTimes,
        32,
        "./tests/json_test_data/hc_2x_ad1.json",
    );
}

#[test]
fn test_2x_hc_ad2() {
    let mut proc = NonlinearProcessor::new();
    proc.change_state(ProcStateTransition::ChangeOrder(
        AntiderivativeOrder::SecondOrder,
    ));
    run_test_case(
        &mut proc,
        OversampleFactor::TwoTimes,
        32,
        "./tests/json_test_data/hc_2x_ad2.json",
    );
}

#[test]
fn test_4x_hc_ad1() {
    let mut proc = NonlinearProcessor::new();
    run_test_case(
        &mut proc,
        OversampleFactor::FourTimes,
        64,
        "./tests/json_test_data/hc_4x_ad1.json",
    );
}

#[test]
fn test_4x_hc_ad2() {
    let mut proc = NonlinearProcessor::new();
    proc.change_state(ProcStateTransition::ChangeOrder(
        AntiderivativeOrder::SecondOrder,
    ));
    run_test_case(
        &mut proc,
        OversampleFactor::FourTimes,
        64,
        "./tests/json_test_data/hc_4x_ad2.json",
    )
}
