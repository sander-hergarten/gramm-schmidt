use std::ops::Sub;

use ndarray::{Array1, Array2, Axis, ScalarOperand, arr2};
use ndarray_linalg::{Lapack, Scalar, normalize};

pub fn gramm_schmidt<A: ScalarOperand + Lapack + Sub>(vectors: Array2<A>) -> Array2<A> {
    let mut m = 0_usize;
    let mut output = Array2::<A>::zeros((vectors.shape()[1], 0));

    for column in vectors.clone().columns() {
        let mut temp_w = column.to_owned();
        if m != 0 {
            temp_w = temp_w.sub(
                (0..m)
                    .map(|p| &output.column(p) * output.column(p).dot(&column))
                    .reduce(|acc, e| acc + e)
                    .unwrap()
                    .view(),
            );
        }

        if &temp_w != &Array1::<A>::zeros(temp_w.len()) {
            println!("{}", temp_w);
            if output.shape()[1] > m {
                output.column_mut(m).assign(&temp_w);
            } else {
                output.push_column(std::mem::replace(
                    &mut temp_w.view(),
                    Array1::<A>::zeros(column.len()).view(),
                ));
            }

            m += 1;
        }
    }

    normalize(output, ndarray_linalg::NormalizeAxis::Column).0
}

// trait Reciprocal {
//     fn recip(self) -> Self;
// }
