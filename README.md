# SPAPT
Signal Processing for Analog Phaser Tomography.

This is the siganl model as well the digital singal processing for the Analog device cn0566 hybrid phaser kit.

## Installation
SPAPT only needs [numpy](https://pypi.org/project/numpy/), [matplotlib](https://pypi.org/project/matplotlib/) and [scipy](https://pypi.org/project/scipy/).

Simply clone SPART form repository.

```bash
git clone https://github.com/TMinRay/SPAPT.git
```

## Usage

To start the simulator
```bash
$ python simradar
```

interactive commands use ` `(space) to differentiate blocks.

## Example Results
To check current platform status.
```SPAPT
$ s p
         current orientation     0.0 degree
         platform speed          0.0 dps
```

### command blocks

| First | second | Parameters | Description |
| --- | --- | --- | --- |
| s | --- | --- | summary of current radar parameters|
| s |  p  | --- | summary only platform parameters|
| s |  t  | --- | summary only transceiver parameters|
| s |  o  | --- | summary only objects(target) parameters|


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
