# MD Simulation

The code in this directory allows for the development of a X-ray or neutron reflectometry profile from a molecular dynamics simulation.
This code was used in the following publication:
```
"McCluskey, A.R., Grant, J., Smith, A.J., Rawle, J.L., Barlow, D.J., Lawrence, M.J., Parker, S.C. & Edler, K.J. (2019). J. Phys. Comm. 3, 075001, https://doi.org/10.1088/2399-6528/ab12a9"
```

Check that everything is working right before use by running the tests:
```
pytest test/test_md_simulation.py
```

## Additional requirements
```
pip install MDAnalysis
```

Added by: [Andrew McCluskey](https://github.com/arm61)

Date: 2019-08-08
