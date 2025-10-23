use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> String {
    "Hello World from rust".to_string()
}

#[pyfunction]
fn add(left: i64, right: i64) -> i64 {
    left + right
}

#[pyfunction]
fn sum_list(numbers: Vec<i64>) -> i64 {
    numbers.iter().sum()
}

#[pymodule]
fn hello_world_rust_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(sum_list, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
