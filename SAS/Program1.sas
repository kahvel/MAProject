data data;
infile '/folders/myfolders/test5_results_2_all.csv';
input ID PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3;
run;
data data;
modify data;
if ID=_NULL_ then remove;
run;


data data;
infile '/folders/myfolders/test5_results_2_psda.csv';
input ID PSDA_P7_Sum_f1 PSDA_P7_Sum_f2 PSDA_P7_Sum_f3 PSDA_O1_Sum_f1 PSDA_O1_Sum_f2 PSDA_O1_Sum_f3 PSDA_O2_Sum_f1 PSDA_O2_Sum_f2 PSDA_O2_Sum_f3 PSDA_P8_Sum_f1 PSDA_P8_Sum_f2 PSDA_P8_Sum_f3 PSDA_P7_h1_f1 PSDA_P7_h1_f2 PSDA_P7_h1_f3 PSDA_O1_h1_f1 PSDA_O1_h1_f2 PSDA_O1_h1_f3 PSDA_O2_h1_f1 PSDA_O2_h1_f2 PSDA_O2_h1_f3 PSDA_P8_h1_f1 PSDA_P8_h1_f2 PSDA_P8_h1_f3 PSDA_P7_h2_f1 PSDA_P7_h2_f2 PSDA_P7_h2_f3 PSDA_O1_h2_f1 PSDA_O1_h2_f2 PSDA_O1_h2_f3 PSDA_O2_h2_f1 PSDA_O2_h2_f2 PSDA_O2_h2_f3 PSDA_P8_h2_f1 PSDA_P8_h2_f2 PSDA_P8_h2_f3 PSDA_P7_h3_f1 PSDA_P7_h3_f2 PSDA_P7_h3_f3 PSDA_O1_h3_f1 PSDA_O1_h3_f2 PSDA_O1_h3_f3 PSDA_O2_h3_f1 PSDA_O2_h3_f2 PSDA_O2_h3_f3 PSDA_P8_h3_f1 PSDA_P8_h3_f2 PSDA_P8_h3_f3 CCA_f1 CCA_f2 CCA_f3;
run;
data data;
modify data;
if ID=_NULL_ then remove;
run;
proc factor data=data rotate=varimax residuals simple nobs=10432 priors=max nfactor=3 out=result;
var PSDA_O1_h1_f1 PSDA_O1_h1_f2 PSDA_O1_h1_f3 CCA_f1 CCA_f2 CCA_f3;
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
Input _NAME_ $ PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3;
list;cards;
FACTOR1 1 0 0 1 0 0
FACTOR2 0 1 0 0 1 0
FACTOR3 0 0 1 0 0 1
;
run;

data TARGET;
Input _NAME_ $ PSDA_h1_f1 PSDA_h2_f1 CCA_f1;
list;cards;
FACTOR1 1 0 0
FACTOR2 0 1 0
FACTOR3 0 0 1
;
run;
/* h2 takes one factor basically.
CCA has very small communality together with multiple PSDA results.
*/
proc factor data=data method=prin residuals simple nobs=10432 priors=one nfactors=1 maxiter=1000 out=result1;
var PSDA_h1_f1 PSDA_h2_f1 CCA_f1;
run;
proc factor data=data method=prin residuals simple nobs=10432 priors=one nfactors=1 maxiter=1000 out=result2;
var PSDA_h1_f2 PSDA_h2_f2 CCA_f2;
run;
proc factor data=data method=prin residuals simple nobs=10432 priors=one nfactors=1 maxiter=1000 out=result3;
var PSDA_h1_f3 PSDA_h2_f3 CCA_f3;
run;
/*Data with target cols, no noise, prinit heywood procrustes h2+CCA, no repeating, classify sum
gave nice ROC curve, but few results*/

/**prinit max gave pretty good result */
/**prin one stable, not much above avg*/



/* varimax+heywook very weird factors */
proc factor data=data method=prin priors=one rotate=procrustes target=target residuals simple nobs=10432 maxiter=1000 nfactors=3 out=result;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3;
run;



/* try out with ouput columns */
proc import
datafile='/folders/myfolders/result_with_cols2.csv'
out=data_new2
dbms=csv
replace;
run;

/*data data_new;
modify data_new;
drop _;
run;*/


data TARGET_NEW;
Input _NAME_ $ PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 t1 t2 t3;
list;cards;
FACTOR1 1 0 0 1 0 0 1 0 0 1 0 0
FACTOR2 0 1 0 0 1 0 0 1 0 0 1 0
FACTOR3 0 0 1 0 0 1 0 0 1 0 0 1
;
run;
proc factor data=data_new rotate=procrustes target=target_new residuals simple nobs=10432 priors=max maxiter=1000 nfactors=3 out=result;
var PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 t1 t2 t3;
run;



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
proc calis data=data residual res all modification plots=all nobs=10432 maxiter=3000 outstat=RESULT_CONF;
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
/*COV
E1 E3 E5=EF1,
E2 E4 E6=EF2;
/*E10 E11 E12*//*,
F1 F2 F3 = 3*0.;
PVAR
F1 F2 F3 = 3*1.;*/
/*VAR PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 CCA_f1 CCA_f2 CCA_f3 /*PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3*/;
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
use RESULT_CONF;
read all var{PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3} into Score where(_type_="SCORE");
print Score;
use data;
read after var{PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3} into X;
XS=standard(X);
F=X*t(Score);
FS=standard(F);
create scores from FS;
append from FS;
close scores;
finish;
run;quit;















data moodud (type=corr);
_type_="CORR";
infile cards missover;
input _NAME_ $ pikkus sormulat saarepik kaal puusaymb rinnaymb ;
datalines;
pikkus   1.0
sormulat 0.846 1.0
saarepik 0.859 0.826 1.0
kaal 0.473 0.376 0.436 1.0
puusaymb 0.398 0.326 0.329 0.762 1.0
rinnaymb 0.301 0.277 0.327 0.730 0.583 1.0
;
proc factor rotate=varimax;
run;


# peakomp;
proc factor data=moodud nfactors=1 method=prin residual;
var pikkus sormulat saarepik kaal puusaymb rinnaymb ;
run;
# peafaktor;
proc factor data=moodud nfactors=1 method=prin residual priors=max; run;
# peafaktor smc;
proc factor data=moodud nfactors=1 method=prin residual priors=smc; run;
# peafaktor iter;
proc factor data=moodud nfactors=1 method=prinit residual priors=max; run;
# peafaktor smc iter;
proc factor data=moodud nfactors=1 method=prinit residual priors=smc; run;


