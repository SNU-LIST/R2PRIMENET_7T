# R2PRIMENET_7T

**R2PRIMENet7T** for 7T x-separation (converts 7T R2star map into 3T R2' map)

## Environment setup

To set up your environment, please use the `r2pnet7T.yaml` configuration file.

## Inference

To run inference, execute `test.py`. 

### Input File

make an 'Data' folder and put input files in it.

The input file must be a `.mat` file, which should contain:
- `r2star` (in Hz, at 7T)
- `Mask`

## License

© Jiye Kim  
@ LIST, Seoul National University

This code and network parameters are made available for academic and research purposes. Please credit the original author when utilizing this work.
