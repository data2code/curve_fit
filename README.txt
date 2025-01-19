usage: fit_file.py [-h] [-i INPUT] [-o OUTPUT] [-m MODEL] [-p PLOT] [-d] [-c CPD] [-x] [-a]

Dose-Response Curve Fitting on Files

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        input data file
  -o OUTPUT, --output OUTPUT
                        output data file
  -m MODEL, --model MODEL
                        model parameter file
  -p PLOT, --plot PLOT  output folder name contains figures
  -d, --debug           print debug message, save debug images into a folder called debug.
  -c CPD, --cpd CPD     a compound ID to process
  -x, --xml             print output within XML tags
  -a, --auto_qc         apply Auto QC, will add additional columns: Mask, Note.


Installation
--------------------

python setup.py install
cp fit_file.py to its final destination folder

Models:
--------------------

It supports two models, A and B. A is for inhibition, where response is expected to go from ~1.0 to ~0.0, as concentrations increases from -inf to inf. For Model B, response is expected to go from ~1.0 to some large values, as concentration increases.

Example
--------------------

./fit_file.py -i 160273.csv -m 160273.json -o out.csv -p image

The normalized concentration and response data points are stored in the 160273.csv file.
Model parameters are specified in the 160273.json file.
The output file out.csv contains the fitted parameters and images are save in the /image folder.

Format
--------------------

Model .json file

{"model":"A", "bounds":[[0.0,0.2],[0.8,1.2],[-3.0,-0.33]], "nMC":8, "outlier":1, "average_points":1}

model="A" means we fit with an inhibition model.
bounds specifies the (lower, upper) bounds for curve bottom, top, and slope. Notice the slope is negative for an inhibition curve.
If the background signal is high, increase bottom accordingly, e.g., [0.0, 0.5].
nMC=8, no need to change, the number of Monte Carlo runs to estimate error bars
outlier=1, allow up to one outlier data points to be excluded
average_points=1, if there are replicate data points, average them first and fit the average response data.

Input data file

CPD,OUTLIER,TOXICITY,AVERAGE_POINTS,MIN_FC,AUC,BOUNDS,CONCENTRATION,RESPONSE
cpd477,1,1,1,0, ,,"50,16.667,5.556,1.8519,0.6173,0.2058,0.0686,0.0229","0.524,0.845,0.977,1.007,1.03,1.054,1.048,0.993"

compound identifiers can be any string, if it starts with "#", the line will be ignored.
OUTLIER,TOXICITY,AVERAGE_POINTS,BOUNDS are already specified in the model .json file. You can overwrite the default if you explicitly
specify the new value here per line.
MIN_FC: for model B, curves with Fold Change (FC) less than MIN_FC will be ignored and fuzzy value will be used for IC50
BOUNDS: if overwrite, the format is ":" separated string, e.g., "0.0:0.2:0.8:1.2" or "0.0:0.2:0.8:1.2:-3.0:-0.33". We recommend not to change slope.
AUC: ignore.
CONCENTRATION, RESPONSE: the data points should be in exactly the same order. If you have a missing data points, you can use empty string: ",,"

Note: we expect RESPONSE values have been pre-normalized, so that the value for DMSO (0uM) should be around 1.0.

Output data file

CPD,param_A,param_B,Fuzzy,param_C,param_D,param_A_std_error,param_B_std_error,param_C_std_error,param_D_std_error,R2,outlier,lock,comment,firstPct,lastPct,outlier
Sensitivity,FC,ignore,AUC,Mask,Note,Concentration,Response
cpd478,0.985,1,,NaN,-0.33,0,0,NaN,inf,0,,1 1 0 0,,NaN,NaN,NaN,0.015,,,Auto Approved,,"50,16.667,5.556,1.8519,0.6173,0.2058,0.0686,0.0229","0.882,0.877,0.915,0.973
c pd482,0.2,1.031,>,50,-0.5229,0,0,69.95,0.1488,0.8065,,1 1 0 0,Missing Bottom,0.9865,0.9338,NaN,0.806,,,Auto Approved,,"50,16.667,5.556,1.8519,0.6173,0.2058,0.0686,0.0229","0.715,0.94,0.931,0.961,0.982,0.986,1.016,1.031"

param_A: bottom
param_B: top
param_C: IC50
param_D: slope

To generate an image for each record, specify the image folder name with -p

Plotting
--------------------
The size of the image is controlled at line 137 in fit_file.py

	plt.plot_one(r, f"{plot}/{r['CPD']}.png", width=400, height=300, popup=True)

The style of the plot is controlled in curve_fit/plot_dr.py


