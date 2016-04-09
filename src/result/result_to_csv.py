
# Take VEP-BCI results (for each steps) data as input an output a CSV file

data_file_name = "test5_results_3_all.txt"
data_file = open(data_file_name)
file_content = data_file.readlines()
data_file.close()
result_file_name = data_file_name[:-4]+".csv"

result_content = "Frequency\n"
frequencies = map(lambda x: x[0], eval(file_content[0])[1][('Sum PSDA', ('P7', 'O1', 'O2', 'P8'))][1])
for i, frequency in enumerate(frequencies):
    result_content += str(i+1) + " " + str(frequency) + "\n"
open(data_file_name[:-6] + "_freq" + data_file_name[-6:], "w").write(result_content)

psda_keys = ["Sum", 1, 2]

result_content = "PSDA_sum_f1 PSDA_sum_f2 PSDA_sum_f3 PSDA_h1_f1 PSDA_h1_f2 PSDA_h1_f3 PSDA_h2_f1 PSDA_h2_f2 PSDA_h2_f3 CCA_f1 CCA_f2 CCA_f3\n"
for i, line in enumerate(file_content):
    results_dict = eval(line)
    result_content += str(i+1) + " "
    for key in psda_keys:
        psda_results = results_dict[1][('Sum PSDA', ('P7', 'O1', 'O2', 'P8'))][key]
        psda_results_dict = dict(psda_results)
        for frequency in frequencies:
            result_content += str(psda_results_dict[frequency]) + " "
    cca_results_dict = dict(results_dict[2][('CCA', ('P7', 'O1', 'O2', 'P8'))])
    for frequency in frequencies:
        result_content += str(cca_results_dict[frequency]) + " "
    result_content = result_content[:-1] + "\n"

result_file = open(result_file_name, "w")
result_file.write(result_content)
result_file.close()

