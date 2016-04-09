data data;
infile '/folders/myfolders/test5_results_2_all.csv';
input ID PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
run;
data data;
modify data;
if ID=_NULL_ then remove;
run;


proc factor data=data rotate=varimax residuals simple nobs=10433 method=prinit priors=smc nfactors=3;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
run;


data TARGET;
Input _NAME_ $ PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
list;cards;
FACTOR1 1 0 0 1 0 0 1 0 0 1 0 0
FACTOR2 0 1 0 0 1 0 0 1 0 0 1 0
FACTOR3 0 0 1 0 0 1 0 0 1 0 0 1
FACTOR4 1 1 1 1 1 1 1 1 1 0 0 0
FACTOR5 0 0 0 0 0 0 0 0 0 1 1 1
;
run;

data TARGET;
Input _NAME_ $ PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 f1 f2 f3;
list;cards;
FACTOR1 1 0 0 1 0 0 1 0 0
FACTOR2 0 1 0 0 1 0 0 1 0
FACTOR3 0 0 1 0 0 1 0 0 1
;
run;

/* h2 takes one factor basically.
CCA has very small communality together with multiple PSDA results.
*/
proc factor data=data_new rotate=procrustes method=prinit heywood target=target residuals simple nobs=10432 priors=max maxiter=1000 nfactors=3;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 f1 f2 f3;
run;


/* varimax+heywook very weird factors */
proc factor data=data heywood method=prinit residuals simple nobs=10432 maxiter=100 rotate=procrustes target=TARGET nfactor=4;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
run;



/* try out with ouput columns */
proc import
datafile='/folders/myfolders/result_with_cols.csv'
out=data_new
dbms=csv
replace;
run;

/*data data_new;
modify data_new;
drop _;
run;*/

data TARGET_NEW;
Input _NAME_ $ PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3 f1 f2 f3;
list;cards;
FACTOR1 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0
FACTOR2 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0
FACTOR3 0 0 1 0 0 1 0 0 1 0 0 1 0 0 1
FACTOR4 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
;
run;

/* priors=max is the simplest, default is PCA*/
proc factor data=data_avg rotate=varimax residuals nobs=10370 simple rotate=procrustes target=TARGET_NEW nfactor=4 priors=max;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
run;

/*10370*/
proc import
datafile='/folders/myfolders/moving_average.csv'
out=data_avg
dbms=csv
replace;
run;



/* Confirmatory */
ods graphics on;
proc calis data=data residual res all modification plots=all nobs=10432 maxiter=3000 outstat=RESULT;
lineqs
/*t1			= a3 F1 + E7,
t2			= b3 F2 + E8,
t3			= c3 F3 + E9,*/
/*PSDA_h1_f1 = a4 F1 + E10,
PSDA_h1_f2 = b4 F2 + E11,
PSDA_h1_f3 = c4 F3 + E12,*/
PSDA_sum_f1 = a1 F1 + E1,
CCA_f1 		= a2 F1 + E2,
PSDA_sum_f2 = b1 F2 + E3,
CCA_f2 		= b2 F2 + E4,
PSDA_sum_f3 = c1 F3 + E5,
CCA_f3 		= c2 F3 + E6;
COV
E1 E3 E5=EF1,
E2 E4 E6=EF2;
/*E10 E11 E12*//*,
F1 F2 F3 = 3*0.;
PVAR
F1 F2 F3 = 3*1.;*/
VAR PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 /*PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3*/;
ods graphics off;


ods graphics on;
proc calis data=data residual res all modification plots=all nobs=10432 /*outstat=RESULT*/;
factor
F1 ===> PSDA_sum_f1 CCA_f1 PSDA_h1_f1,
F2 ===> PSDA_sum_f2 CCA_f2 PSDA_h1_f2,
F3 ===> PSDA_sum_f3 CCA_f3 PSDA_h1_f3;
COV
F1 F2 F3 = 3*0.;
PVAR
F1 F2 F3 = 3*1.;
VAR PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 CCA_f1 CCA_f2 CCA_f3;
ods graphics off;

proc iml;
start;
use RESULT;
read all var{PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3} into Score where(_type_="SCORE");
print Score;
use data;
read after var{PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3} into X;
XS=standard(X);
F=X*t(Score);
FS=standard(F);
create scores from FS;
append from FS;
close scores;
finish;
run;quit;




