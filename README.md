# MAProject

This repository contains the results of two projects done by me (Anti Ingel, antiingel@gmail.com) in the courses at University of Tartu and possibly some ongoing research. The projects build upon the results of my thesis ([Control a Robot via VEP using Emotiv EPOC](http://comserv.cs.ut.ee/forms/ati_report/downloader.php?file=FF16189169B7081D7F8121C4E2736D6C8384C450), code can be found [here](https://github.com/kahvel/VEP-BCI)) which was about visual evoked potential based brain-computer interface (BCI). For a very brief overview of the thesis see this [poster](https://github.com/kahvel/VEP-BCI/blob/master/docs/images/poster.pdf). In these projects, the data recorded and processed by the BCI is used.

The first project was done in the course Multidimensional Analysis and it is about exploratory factor analysis of the features extracted by the BCI. The resulting report (in Estonian) can be found [here](). The factor analysis was able to divide the extracted features into groups so that each group corresponded to exactly one command that could be sent to the BCI, which suggests that there is indeed some similarity between the extracted features that correspond to the same command. To learn more about the extracted features and commands of the BCI, please refer to my thesis or the posters.

The second project got inspiration from the first and was done in the course Data Mining. It is about using machine learning techniques to predict the command using the features extracted by the BCI (and other features derived from these). See this [poster]() for the summary of this project. In this project several machine learning methods were tested and finally it was concluded that support vector machine together with boosting of decision tree stumps gave the best results.

## Repository structure

* R - contains the R files of Multivariate Analysis project. Focuses on exploratory analysis of the data.
* SAS - contains the SAS files of Multivariate Analysis project. Focuses exlusively on exploratory factor analysis.
* data - contains the recorded EEG signal, extracted features (file name contains "results"), the used command frequencies (file name contains "freq") and the expected commands (file name contains "targets").
* Poster - contains Tex files of the Data Mining project poster.
* src - contains Python files for converting the EEG data given by the BCI to R and SAS readable files and all the files of Data mining project that were used in testing and training the machine learning models and analysing the results.
* tex - contains Tex files of the Multivariate Analysis project report.

## Data

This repository contains all the data recorded and processed with the BCI for the test subject 5. EEG data contains raw EEG signal from the Emotiv EPOC device. Note that only the sensors P7, O1, O2, P8 contain useful information, since other sensors were not used. The time interval between observations (packets) is 1/128 seconds. The extracted features data contains the features extracted by the BCI using window length of 256 EEG packets (= 2 seconds). The features are encoded as follows:

* CCA - means that the feature was extracted by canonical correlation analysis.
* PSDA - means that the feature was extracted by power spectral density analysis.
* f1 - the command for which the feature was extracted (f1-f3, since there were three commands).
* h1 - harmonic for which the feature was extracted (only used with PSDA).
* sum - all the features of different harmonics for ome command summed up (only used with PSDA).

Target data contains information about the expected commands, the ranges are encoded with the EEG packet number. And finally, frequency data contains information about which frequencies were used for the targets (commands) in Hz.
