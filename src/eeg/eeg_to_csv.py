
# Take EEG data recorded with VEP-BCI as input and output a CSV file

data_file_name = "test5.txt"
data_file = open(data_file_name)
file_content = data_file.read().replace("\n", "")
data_file.close()
split_content = file_content.split(";")
list_of_trials = eval(split_content[0])
result_file_name = data_file_name[:-4]+".csv"

SENSORS = ("AF3", "F7", "F3", "FC5","T7", "P7", "O1", "O2", "P8", "T8", "FC6","F4", "F8", "AF4")

for i in range(len(list_of_trials)):
    file_content = "Frequency\n"
    for key in list_of_trials[i]["TargetFreqs"]:
        file_content += str(key) + " " + str(list_of_trials[i]["TargetFreqs"][key]) + "\n"

    frequencies_file = open(result_file_name[:-4] + "_freq_" + str(i+1) + result_file_name[-4:], "w")
    frequencies_file.write(file_content)
    frequencies_file.close()

    file_content = " ".join(SENSORS) + "\n"
    for j, packet in enumerate(list_of_trials[i]["Packets"]):
        file_content += str(j+1) + " "
        for sensor in SENSORS:
            file_content += str(packet[sensor]) + " "
        file_content = file_content[:-1] + "\n"

    result_file = open(result_file_name[:-4] + "_" + str(i+1) + result_file_name[-4:], "w")
    result_file.write(file_content)
    result_file.close()
