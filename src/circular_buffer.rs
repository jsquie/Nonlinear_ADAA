use num_traits::Float;

#[derive(Debug)]
pub struct CircularBuffer<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Float,
{
    data: Vec<T>,
    pos: usize,
    size: usize,
}

impl<T> CircularBuffer<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Float + From<f32>,
{
    pub fn new(initial_size: usize) -> Self {
        CircularBuffer {
            data: vec![0.0_f32.into(); initial_size],
            pos: 0,
            size: initial_size,
        }
    }

    pub fn reset(&mut self) {
        self.data.iter_mut().for_each(|x| *x = 0.0_f32.into());
        self.pos = 0;
    }

    #[inline]
    fn push(&mut self, val: T) {
        self.data[self.pos] = val;
    }

    #[inline]
    fn decrement_pos(&mut self) {
        self.pos = if self.pos == 0 {
            self.size - 1
        } else {
            self.pos - 1
        };
    }

    #[inline]
    pub fn delay(&mut self, val: T) -> T {
        let res = self.data[self.pos];
        self.push(val);
        self.decrement_pos();
        res
    }

    #[inline]
    fn dot(&self, other: &[T]) -> T {
        assert!(self.size == other.len());
        let p = self.pos;
        let o = (self.size as usize) - p;
        // f_h = data[p..] dot other[..o_pos]
        // s_h = data[o_pos..] dot other[o_pos..]

        let fh: T = self.data[p..]
            .iter()
            .zip(other[..o].iter())
            .fold(0.0_f32.into(), |acc, (a, b)| acc + (*a * *b));
        let sh: T = self.data[..p]
            .iter()
            .zip(other[o..].iter())
            .fold(0.0_f32.into(), |acc, (a, b)| acc + (*a * *b));
        fh + sh
    }

    #[inline]
    pub fn convolve(&mut self, val: T, coeffs: &[T]) -> T {
        let res: T;
        self.push(val);
        res = self.dot(&coeffs);
        self.decrement_pos();
        res
    }
}

#[cfg(test)]
mod tests {
    // use crate::circular_buffer;

    use super::CircularBuffer;

    #[test]
    fn create_f32() {
        let new = CircularBuffer::<f32>::new(1);
        assert_eq!(new.pos, 0);
        assert_eq!(new.size, 1);
        assert_eq!(new.data, vec![0.0]);
    }

    #[test]
    fn create_f64() {
        let new = CircularBuffer::<f64>::new(1);
        assert_eq!(new.pos, 0);
        assert_eq!(new.size, 1);
        assert_eq!(new.data, vec![0.0]);
    }

    #[test]
    fn push_sucess() {
        let mut new = CircularBuffer::<f32>::new(1);
        new.push(1.);
        assert_eq!(new.data[0], 1.);
    }

    #[test]
    fn ptr_dec() {
        let mut new = CircularBuffer::<f32>::new(2);
        assert_eq!(new.pos, 0);
        new.decrement_pos();
        assert_eq!(new.pos, 1);
        new.decrement_pos();
        assert_eq!(new.pos, 0);
    }

    #[test]
    fn simple_dot() {
        let mut a = CircularBuffer::<f32>::new(3);
        let mut b = CircularBuffer::<f32>::new(3);

        a.data = vec![0., 1., 2.];
        b.data = vec![0., 1., 2.];

        assert_eq!(a.dot(&b.data), 5.);
    }

    #[test]
    fn test_conv_01234_012() {
        let signal = vec![0., 1., 2., 3., 4., 0., 0.];
        let mut buf = CircularBuffer::<f32>::new(3);
        let coefs = vec![0., 1., 2.];

        let res = signal
            .iter()
            .map(|x| buf.convolve(*x, &coefs))
            .collect::<Vec<_>>();
        assert_eq!(res, vec![0., 0., 1., 4., 7., 10., 8.])
    }

    #[test]
    fn conv_sin_filter() {
        let sig: Vec<f64> = (0..30)
            .map(|x| (((x as f64) * std::f64::consts::PI * 2.0) / 44100.0).sin())
            .collect();

        let mut buf = CircularBuffer::<f64>::new(10);
        let coefs: [f64; 10] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];

        let res: Vec<f32> = sig
            .iter()
            .map(|x| buf.convolve(*x, &coefs) as f32)
            .collect();

        let expected_result = vec![
            0.00000000e+00,
            0.00000000e+00,
            1.42475857e-04,
            5.69903424e-04,
            1.42475855e-03,
            2.84951708e-03,
            4.98665483e-03,
            7.97864762e-03,
            1.19679712e-02,
            1.70971015e-02,
            2.35085141e-02,
            2.99199262e-02,
            3.63313377e-02,
            4.27427485e-02,
            4.91541584e-02,
            5.55655673e-02,
            6.19769751e-02,
            6.83883817e-02,
            7.47997868e-02,
            8.12111904e-02,
            8.76225924e-02,
            9.40339926e-02,
            1.00445391e-01,
            1.06856787e-01,
            1.13268181e-01,
            1.19679573e-01,
            1.26090962e-01,
            1.32502349e-01,
            1.38913733e-01,
            1.45325114e-01,
        ];

        assert_eq!(res, expected_result);
    }

    #[test]
    fn delay_5_samples() {
        let sig: Vec<f32> = (1..10).map(|x| x as f32).collect();
        let mut delay_buf = CircularBuffer::<f32>::new(5);

        let delay_one = delay_buf.delay(sig[0]);
        assert_eq!(delay_one, 0.);
        assert_eq!(delay_buf.data, vec![1., 0., 0., 0., 0.]);

        let delay_two = delay_buf.delay(sig[1]);
        assert_eq!(delay_two, 0.);
        assert_eq!(delay_buf.data, vec![1., 0., 0., 0., 2.]);

        let delay_three = delay_buf.delay(sig[2]);
        assert_eq!(delay_three, 0.);
        assert_eq!(delay_buf.data, vec![1., 0., 0., 3., 2.]);

        let delay_four = delay_buf.delay(sig[3]);
        assert_eq!(delay_four, 0.);
        assert_eq!(delay_buf.data, vec![1., 0., 4., 3., 2.]);

        let delay_five = delay_buf.delay(sig[4]);
        assert_eq!(delay_five, 0.);
        assert_eq!(delay_buf.data, vec![1., 5., 4., 3., 2.]);

        let delay_six = delay_buf.delay(sig[5]);
        assert_eq!(delay_six, sig[0]);
        assert_eq!(delay_buf.data, vec![6., 5., 4., 3., 2.]);

        let delay_seven = delay_buf.delay(sig[6]);
        assert_eq!(delay_seven, sig[1]);
        assert_eq!(delay_buf.data, vec![6., 5., 4., 3., 7.]);
    }

    #[test]
    fn delay_list() {
        let sig: Vec<_> = (1..10).map(|x| x as f32).collect();
        let mut delay_buf = CircularBuffer::<f32>::new(5);

        let result = sig
            .into_iter()
            .map(|x| delay_buf.delay(x))
            .collect::<Vec<f32>>();

        assert_eq!(
            result,
            vec![vec![0.; 5], (1..5).map(|x| x as f32).collect()]
                .into_iter()
                .flatten()
                .collect::<Vec<f32>>()
        )
    }

    #[test]
    fn test_convolve_filter_taps() {
        let filter_taps = vec![
            -0.0064715474097890545,
            0.006788724784527351,
            -0.007134125572070907,
            0.007511871271766723,
            -0.007926929217098087,
            0.00838534118242672,
            -0.00889453036904902,
            0.009463720022395613,
            -0.010104514094437885,
            0.010831718180021,
            -0.011664525313602769,
            0.012628270948224513,
            -0.013757103575462731,
            0.015098181413680897,
            -0.01671851963595936,
            0.01871667093508393,
            -0.021243750540180146,
            0.024543868940610197,
            -0.0290386730354654,
            0.035524608815134716,
            -0.045708348639099484,
            0.06402724397938601,
            -0.10675158913607562,
            0.32031404953367254,
            0.32031404953367254,
            -0.10675158913607562,
            0.06402724397938601,
            -0.045708348639099484,
            0.035524608815134716,
            -0.0290386730354654,
            0.024543868940610197,
            -0.021243750540180146,
            0.01871667093508393,
            -0.01671851963595936,
            0.015098181413680897,
            -0.013757103575462731,
            0.012628270948224513,
            -0.011664525313602769,
            0.010831718180021,
            -0.010104514094437885,
            0.009463720022395613,
            -0.00889453036904902,
            0.00838534118242672,
            -0.007926929217098087,
            0.007511871271766723,
            -0.007134125572070907,
            0.006788724784527351,
            -0.0064715474097890545,
        ];

        let expected_result = vec![
            0.0,
            -0.0064715474097890545,
            -0.006154370035050758,
            -0.012971318232383369,
            0.013609794481206961,
            -0.014305563389777363,
            0.015067096563530714,
            -0.015904635655489836,
            0.016830682831577737,
            -0.01786066515679372,
            0.019013850058332067,
            -0.020314631236874423,
            0.021794374861081975,
            -0.02349413761982201,
            0.02546878710742898,
            -0.027793467534985756,
            0.030574175904207905,
            -0.03396596757789036,
            0.0382063806655017,
            -0.04368218677478544,
            0.05107886956603451,
            -0.06177515011522625,
            0.07918437314659119,
            -0.11582214709460205,
            0.29889260319967936,
            0.6406873811927908,
            1.4948186585322873,
            0.8114662143082524,
            -0.23790862808855429,
            0.13618964347509377,
            -0.09511450132249442,
            0.07304034931508355,
            -0.05927203176535595,
            0.04986077667655423,
            -0.043016429386331934,
            0.037811154947013974,
            -0.03371629965597902,
            0.030408608038341743,
            -0.027679294143541935,
            0.025387480397489004,
            -0.02343465367520419,
            0.02174984637358284,
            -0.02028063260757145,
            0.01898744051151552,
            -0.01783983795939171,
            0.01681403638485071,
            -0.015891170679831722,
            0.015056087455685704,
            -0.014296474556947074,
            0.007423079534003944,
            -0.019414642229367163,
        ];

        let sig = vec![vec![0., 1., 2., 3.], vec![0.; expected_result.len() - 4]]
            .into_iter()
            .flatten();

        let mut buff = CircularBuffer::<f64>::new(filter_taps.len());

        let result: Vec<f32> = sig
            .into_iter()
            .map(|x| buff.convolve(x, &filter_taps) as f32)
            .collect();

        assert_eq!(
            result,
            expected_result
                .into_iter()
                .map(|x| x as f32)
                .collect::<Vec<f32>>()
        );
    }
}
