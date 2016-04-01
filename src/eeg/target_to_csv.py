
# Take EEG data recorded with VEP-BCI as input and output a CSV file

data_file_name = "test5.txt"
data_file = open(data_file_name)
file_content = data_file.read().replace("\n", "")
data_file.close()
split_content = file_content.split(";")
list_of_trials = eval(split_content[2])
result_file_name = data_file_name[:-4] + "_targets" + ".csv"

packet_lists = eval(split_content[0])
lengths = map(lambda x: len(x["Packets"]), packet_lists)

for i in range(len(list_of_trials)):
    file_content = "Start Stop Target\n"
    for j, target_data in enumerate(list_of_trials[i]):
        if j == len(list_of_trials[i]) - 1:
            stop = lengths[i]
        else:
            stop = list_of_trials[i][j+1][1]
        file_content += str(j+1) + " " + str(target_data[1]+1) + " " + str(stop) + " " + str(target_data[0]) + "\n"

    result_file = open(result_file_name[:-4] + "_" + str(i+1) + result_file_name[-4:], "w")
    result_file.write(file_content)
    result_file.close()
