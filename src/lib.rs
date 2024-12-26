use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind::InvalidInput, Seek, SeekFrom};

struct LineReader {
    reader: BufReader<File>,
    line: String,
    reached_end: bool,
    got_error: bool,
    bytes_read: usize,
    cursor_offset: usize,
    line_start_offset: usize,
}

impl LineReader {
    fn new(file_path: &str, offset: usize) -> Result<Self, Error> {
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        reader.seek(SeekFrom::Start(offset as u64))?;
        Ok(Self {
            reader,
            line: String::new(),
            reached_end: false,
            got_error: false,
            bytes_read: 0,
            cursor_offset: offset,
            line_start_offset: offset,
        })
    }
    fn read_line(&mut self) -> Result<(), Error> {
        self.line.clear();
        self.line_start_offset = self.cursor_offset;
        match self.reader.read_line(&mut self.line) {
            Ok(bytes_read) => {
                self.bytes_read = bytes_read;
                self.cursor_offset += bytes_read;
                if bytes_read == 0 {
                    self.reached_end = true;
                }
                Ok(())
            }
            Err(e) => {
                self.bytes_read = 0;
                self.got_error = true;
                self.reached_end = true;
                Err(e)
            }
        }
    }
    fn take_line(&mut self) -> String {
        std::mem::take(&mut self.line)
    }
}

struct ConfigReader {
    reader: LineReader,
    save_lines: bool,
}

impl ConfigReader {
    fn new(filepath: &str, offset: usize, save_lines: bool) -> Result<Self, Error> {
        Ok(Self {
            reader: LineReader::new(filepath, offset)?,
            save_lines,
        })
    }
}

impl Iterator for ConfigReader {
    type Item = Result<(usize, usize, Vec<String>), Error>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut lines = Vec::new();
        if self.reader.reached_end || self.reader.got_error {
            return None;
        }
        while !self.reader.line.starts_with('t') {
            if let Err(e) = self.reader.read_line() {
                return Some(Err(e));
            }
            if self.reader.reached_end || self.reader.got_error {
                return None;
            }
        }
        let config_start = self.reader.line_start_offset;
        if self.save_lines {
            lines.push(self.reader.take_line());
        }
        if let Err(e) = self.reader.read_line() {
            return Some(Err(e));
        }
        while !self.reader.line.starts_with('t') && !self.reader.reached_end {
            if self.save_lines {
                lines.push(self.reader.take_line());
            }
            if let Err(e) = self.reader.read_line() {
                return Some(Err(e));
            }
        }
        Some(Ok((config_start, self.reader.line_start_offset, lines)))
    }
}

#[derive(Debug)]
pub struct Configuration {
    time: u64,
    cbox: Vec<f64>,
    cenergy: Vec<f64>,
    nucleotides: Vec<Vec<f64>>,
}

impl Configuration {
    fn get_header<'a>(
        lines: &'a [String],
        index: usize,
        start_with: &str,
        header_type: &str,
    ) -> Result<&'a str, Error> {
        let line = lines.get(index).ok_or(Error::new(
            InvalidInput,
            format!("Missing {} header line", header_type),
        ))?;
        if !line.starts_with(start_with) {
            return Err(Error::new(
                InvalidInput,
                format!(
                    "line {} does not start with {}: {}",
                    header_type, start_with, line
                ),
            ));
        }
        let split = line.split('=').nth(1).ok_or(Error::new(
            InvalidInput,
            format!("Invalid {} header format: {}", header_type, line),
        ))?;
        Ok(split.trim())
    }

    fn parse_values<T: std::str::FromStr>(
        values: &str,
        count: usize,
        name: &str,
    ) -> Result<Vec<T>, Error> {
        let parsed = values
            .split(' ')
            .map(|s| {
                s.parse().map_err(|_| {
                    Error::new(
                        InvalidInput,
                        format!("Invalid {} header value \"{}\"", name, values),
                    )
                })
            })
            .collect::<Result<Vec<T>, Error>>()?;
        if parsed.len() != count {
            return Err(Error::new(
                InvalidInput,
                format!("Invalid {} header values \"{}\"", name, values),
            ));
        }
        Ok(parsed)
    }

    fn from_lines(lines: Vec<String>) -> Result<Self, Error> {
        let time_str = Self::get_header(&lines, 0, "t", "time")?;
        let time = time_str.parse().map_err(|_| {
            Error::new(
                InvalidInput,
                format!("Invalid time header value \"{}\"", time_str),
            )
        })?;

        let cbox_str = Self::get_header(&lines, 1, "b", "box")?;
        let cbox: Vec<f64> = Self::parse_values::<f64>(cbox_str, 3, "box")?;

        let cenergy_str = Self::get_header(&lines, 2, "E", "energy")?;
        let cenergy: Vec<f64> = Self::parse_values::<f64>(cenergy_str, 3, "energy")?;

        let nucleotides = lines
            .into_iter()
            .skip(3)
            .map(|line| Self::parse_values(line.trim(), 15, "nucleotide"))
            .collect::<Result<Vec<Vec<f64>>, Error>>()?;
        Ok(Self {
            time,
            cbox,
            cenergy,
            nucleotides,
        })
    }
}

pub fn read_confs(
    file_path: &str,
    offset: usize,
    limit: usize,
) -> Result<Vec<(usize, Configuration)>, Error> {
    let reader = ConfigReader::new(file_path, offset, true)?;
    let mut results = reader
        .take(limit)
        .enumerate()
        .par_bridge()
        .map(|(index, result)| match result {
            Err(e) => (index, Err(e)),
            Ok((_, end_offset, lines)) => match Configuration::from_lines(lines) {
                Ok(conf) => (index, Ok((end_offset, conf))),
                Err(e) => (index, Err(e)),
            },
        })
        .collect::<Vec<_>>();
    results.sort_by_key(|(index, _)| *index);
    results.into_iter().map(|(_, result)| result).collect()
}

pub fn read_offsets(file_path: &str, offset: usize, limit: usize) -> Result<Vec<usize>, Error> {
    let reader = ConfigReader::new(file_path, offset, false)?;
    reader
        .take(limit)
        .map(|result| result.map(|(_, end_offset, _)| end_offset))
        .collect::<Result<Vec<usize>, Error>>()
}

#[pyfunction]
fn read_configurations<'py>(
    py: Python<'py>,
    file_path: &str,
    offset: usize,
    limit: usize,
) -> PyResult<(
    Vec<usize>,
    Vec<(
        u64,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
    )>,
)> {
    match read_confs(file_path, offset, limit) {
        Err(e) => {
            if e.kind() == InvalidInput {
                return Err(PyValueError::new_err(e.to_string()));
            }
            return Err(PyIOError::new_err(e.to_string()));
        }
        Ok(configs) => {
            let end_offsets = configs
                .iter()
                .map(|(end_offset, _)| end_offset.to_owned())
                .collect::<Vec<usize>>();

            let confs = configs
                .into_iter()
                .map(|(_, conf)| {
                    let np_box = PyArray1::from_vec(py, conf.cbox);
                    let np_energy = PyArray1::from_vec(py, conf.cenergy);
                    let np_nucleotides = PyArray2::from_vec2(py, &conf.nucleotides)?;
                    Ok((conf.time, np_box, np_energy, np_nucleotides))
                })
                .collect::<Result<
                    Vec<(
                        u64,
                        Bound<'py, PyArray1<f64>>,
                        Bound<'py, PyArray1<f64>>,
                        Bound<'py, PyArray2<f64>>,
                    )>,
                    PyErr,
                >>()?;

            Ok((end_offsets, confs))
        }
    }
}

#[pyfunction]
fn read_indicies(file_path: &str, offset: usize, limit: usize) -> PyResult<Vec<usize>> {
    read_offsets(file_path, offset, limit).map_err(|e| PyIOError::new_err(e.to_string()))
}

pub fn dumps_conf(
    time: u64,
    box_array: numpy::ndarray::ArrayView1<'_, f64>,
    energy_array: numpy::ndarray::ArrayView1<'_, f64>,
    nucleotides_array: numpy::ndarray::ArrayView2<'_, f64>,
) -> String {
    let box_values = box_array
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let energy_values = energy_array
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    let header = format!("t = {}\nb = {}\nE = {}", time, box_values, energy_values);

    let mut lines = nucleotides_array
        .axis_iter(numpy::ndarray::Axis(0))
        .map(|line| {
            line.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>();

    lines.insert(0, header);
    lines.push(String::new());

    lines.join("\n")
}

#[pyfunction]
fn dumps_configurations(configs: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let configs: Vec<(
        u64,
        Bound<'_, PyArray1<f64>>,
        Bound<'_, PyArray1<f64>>,
        Bound<'_, PyArray2<f64>>,
    )> = configs.extract()?;

    let refs = configs
        .into_iter()
        .map(|(time, np_box, np_energe, np_nucleotides)| {
            (
                time,
                np_box.readonly(),
                np_energe.readonly(),
                np_nucleotides.readonly(),
            )
        })
        .collect::<Vec<_>>();

    let arrays = refs
        .iter()
        .map(|(time, box_ref, energy_ref, nucleotides_ref)| {
            (
                *time,
                (*box_ref).as_array(),
                (*energy_ref).as_array(),
                (*nucleotides_ref).as_array(),
            )
        })
        .collect::<Vec<_>>();

    let serialized = arrays
        .par_iter()
        .map(|(time, box_array, energy_array, nucleotides_array)| {
            dumps_conf(*time, *box_array, *energy_array, *nucleotides_array)
        })
        .collect::<Vec<_>>();

    Ok(serialized)
}

#[pymodule]
fn oxdna_trajectory_reader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_configurations, m)?)?;
    m.add_function(wrap_pyfunction!(read_indicies, m)?)?;
    m.add_function(wrap_pyfunction!(dumps_configurations, m)?)?;
    Ok(())
}
