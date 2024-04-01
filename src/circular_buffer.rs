use crate::nih_debug_assert_eq;
use ndarray::{s, Array1};
use num_traits::Float;

#[derive(Debug)]
struct CircularBuffer<T>
where
    T: Float + 'static,
{
    data: Array1<T>,
    pos: usize,
    size: i32,
}

impl<T> CircularBuffer<T>
where
    T: Float + 'static,
{
    pub fn new(initial_size: usize) -> Self {
        CircularBuffer {
            data: Array1::<T>::zeros(initial_size),
            pos: 0,
            size: initial_size as i32,
        }
    }

    fn push(&mut self, val: T) {
        self.data[self.pos] = val;
    }

    fn decrement_pos(&mut self) {
        self.pos = if self.pos == 0 {
            (self.size - 1) as usize
        } else {
            self.pos - 1
        };
    }

    pub fn delay(&mut self, val: T) -> T {
        let output = self.data[self.pos];
        self.push(val);
        self.decrement_pos();
        output
    }

    fn dot(&self, other: &Array1<T>) -> T {
        nih_debug_assert_eq!(self.data.shape(), other.shape());
        let p = self.pos;
        let o_pos = (self.size as usize) - p;
        self.data.slice(s![p..]).dot(&other.slice(s![..o_pos]))
            + self.data.slice(s![..p]).dot(&other.slice(s![o_pos..]))
    }

    pub fn convolve(&mut self, val: T, coeffs: &Array1<T>) -> T {
        nih_debug_assert_eq!(self.data.shape(), coeffs.shape());
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
    use ndarray::{array, Array1};

    use super::CircularBuffer;

    #[test]
    fn create_f32() {
        let new = CircularBuffer::<f32>::new(1);
        assert_eq!(new.pos, 0);
        assert_eq!(new.size, 1);
        assert_eq!(new.data, Array1::<f32>::zeros(1));
    }

    #[test]
    fn create_f64() {
        let new = CircularBuffer::<f64>::new(1);
        assert_eq!(new.pos, 0);
        assert_eq!(new.size, 1);
        assert_eq!(new.data, Array1::<f64>::zeros(1));
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

        a.data = array!(0., 1., 2.);
        b.data = array!(0., 1., 2.);

        assert_eq!(a.dot(&b.data), 5.);
    }

    #[test]
    fn test_conv_01234_012() {
        let signal = array!(0., 1., 2., 3., 4., 0., 0.);
        let mut buf = CircularBuffer::<f32>::new(3);
        let coefs = array!(0., 1., 2.);

        let res = signal.map(|x| buf.convolve(*x, &coefs));
        assert_eq!(res, array![0., 0., 1., 4., 7., 10., 8.])
    }

    #[test]
    fn conv_sin_filter() {
        let sig = Array1::from_vec(
            (0..30)
                .map(|x| (((x as f64) * std::f64::consts::PI * 2.0) / 44100.0).sin())
                .collect(),
        );

        let mut buf = CircularBuffer::<f64>::new(10);
        let coefs = Array1::from_vec((0..10).map(|x| x as f64).collect());

        let res = sig.mapv(|x| buf.convolve(x, &coefs) as f32);

        let expected_result = array!(
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
        );

        assert_eq!(res, expected_result);
    }

    #[test]
    fn delay_5_samples() {
        let sig = Array1::from_vec((1..10).map(|x| x as f32).collect());
        let mut delay_buf = CircularBuffer::<f32>::new(5);

        let delay_one = delay_buf.delay(sig[0]);
        assert_eq!(delay_one, 0.);
        assert_eq!(delay_buf.data, array!(1., 0., 0., 0., 0.));

        let delay_two = delay_buf.delay(sig[1]);
        assert_eq!(delay_two, 0.);
        assert_eq!(delay_buf.data, array!(1., 0., 0., 0., 2.));

        let delay_three = delay_buf.delay(sig[2]);
        assert_eq!(delay_three, 0.);
        assert_eq!(delay_buf.data, array!(1., 0., 0., 3., 2.));

        let delay_four = delay_buf.delay(sig[3]);
        assert_eq!(delay_four, 0.);
        assert_eq!(delay_buf.data, array!(1., 0., 4., 3., 2.));

        let delay_five = delay_buf.delay(sig[4]);
        assert_eq!(delay_five, 0.);
        assert_eq!(delay_buf.data, array!(1., 5., 4., 3., 2.));

        let delay_six = delay_buf.delay(sig[5]);
        assert_eq!(delay_six, sig[0]);
        assert_eq!(delay_buf.data, array!(6., 5., 4., 3., 2.));

        let delay_seven = delay_buf.delay(sig[6]);
        assert_eq!(delay_seven, sig[1]);
        assert_eq!(delay_buf.data, array!(6., 5., 4., 3., 7.));
    }
}
