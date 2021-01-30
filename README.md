# RPSI
**R**ock **P**hysics **S**eismic **I**nversion is a python realization 
of seismic inversion direct in petro-elastic field. 
This algorithm minimizes misfit 
between observed and modelled seismic data by optimizing 
rockphysics components.

## Project structure


## Pipeline
### 1. Rock physics modelling
The first stage is rockphysics modelling. Its goal is to 
calculate seismic attributes (Vp, Vs, density) from the given rockphysics ones. 
Any new model could be added. Rock physics code lives in 
[fmodelling/rock_physics](https://github.com/pnkrtm/rpsi/tree/main/fmodeling/rock_physics). 
Available models:
* BGTL  *(Biot-Gassmann-Lee)*
* DEM *(differential effective medium)*
* Gassmann
* Kuster-Toksoz
* Voigt-Reuss-Hill

### 2. Seismic forward modelling
Seismic forward modelling calculates ray kinematics, ray dynamics 
and seismograms from seismic attributes. This stage consists of two substages: 
raytracing and AVO calculation.

#### 2.1. Ray tracing
#### 2.2. AVO calculation

### 3. Attributes optimization


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

If you have any questions, please feel free to contact me 
via e-mail: *penkin.msu@gmail.com*

## License
[MIT](https://choosealicense.com/licenses/mit/)
