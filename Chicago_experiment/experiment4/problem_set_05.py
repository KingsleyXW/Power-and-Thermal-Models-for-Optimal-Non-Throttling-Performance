import csv

print('PROBLEM SET 5\n')

# PROBLEM 1
def read_csv(filepath, encoding='utf-8-sig'):
    """
    Reads a CSV file, parsing row values per the provided delimiter. Returns a list of lists,
    wherein each nested list represents a single row from the input file.

    Parameters:
        filepath (str): The location of the file to read
        encoding (str): name of encoding used to decode the file

    Returns:
        list: nested "row" lists
    """

    with open(filepath, 'r', encoding=encoding) as file_obj:
        data = []
        reader = csv.reader(file_obj)
        for row in reader:
            data.append(row)

        return data

# PROBLEM 2
def clean_data(data):
    """
    Strips strings of leading and trailing whitespaces, converts numbers to integers,
    splits strings on a delimiter and converts the case of a string from the split string.

    Parameters:
        data (list): A list of lists, each representing a party's election results
    Returns:
        None
    """

    for info in data[1:]:
        info[0] = info[0].strip()
        info[1] = int(info[1])
        lst = info[2].split('/')
        info[2] = lst[0]
        info.append(lst[1].title())
    data[0][2] = 'Position'
    data[0].append('Party Leader')
# PROBLEM 3
def get_leader_province(leader_provinces, party_info):
    """
    Adds the home province of a party leader to the list containing their party's election info.

    Parameters:
        leader_provinces (list): A list containing the names of party leaders and their home provinces.
        party_info (list): A list containing a party's information from a specific election.
    Returns:
        None
    """

    for leader_pro in leader_provinces[1:]:

        if leader_pro[0] == party_info[3]:
            party_info.append(leader_pro[1])


# PROBLEM 4
def get_seats_by_province(election_data, province):
    """
    Gets the total seats for parties whose leaders are from a specific province.

    Parameters:
        election_data (list): A list of lists containing information from a specific election.
        province (str): The name of a province.
    Returns:
        (int): The total number of seats for parties whose leaders are from the given province.
    """

    province_seats = 0
    for ele in election_data[1:]:
        if province.lower() == ele[-1].lower():
            province_seats += ele[1]
    return province_seats
# PROBLEM 5
def get_seat_difference(prev_election_data, party_info):
    """
    Adds the difference in seats between two elections for a party to that party's info list.

    Parameters:
        prev_election_data (list): A list of lists containing information from a specific election.
        party_info (list): A list containing a party's information from a specific election.
    Returns:
        (int): An integer value of the seat difference for a party between the current and previous election.
    """

    for prev in prev_election_data[1:]:
        if party_info[0] == prev[0]:
           return party_info[1] - prev[1]
# PROBLEM 6
def write_csv(filepath, data, headers=None, encoding='utf-8', newline=''):
    """
    Writes data to a target CSV file. Column headers are written as the first
    row of the CSV file if optional headers are specified.

    Parameters:
        filepath (str): path to target file (if file does not exist it will be created)
        data (list | tuple): sequence to be written to the target file
        headers (seq): optional header row list or tuple
        encoding (str): name of encoding used to encode the file
        newline (str): specifies replacement value for newline '\n'
                       or '\r\n' (Windows) character sequences

    Returns:
        None
    """

    with open(filepath, 'w', encoding=encoding, newline=newline) as file_obj:
        writer = csv.writer(file_obj)
        if headers:
            writer.writerow(headers)
            for row in data:
                writer.writerow(row)
        else:
            writer.writerows(data)

def main():
    """
    Serves as the main point of entry point of the program.
    """

    #PROBLEM 1.2
    print('\nProblem 1:\n')
    filepath1 = './election_data_2014.csv'
    filepath2 = './election_data_2019.csv'
    election_data_2014 = read_csv(filepath1)
    election_data_2019 = read_csv(filepath2)
    print(election_data_2014)
    print(election_data_2019)
    
    #PROBLEM 2.2
    print("\nProblem 2:\n")
    clean_data(election_data_2014)
    clean_data(election_data_2019)
    print(election_data_2014)
    print(election_data_2019)

    # PROBLEM 3.2
    print('\nProblem 3:\n')
    filepath3 = './leader_home_provinces.csv'
    home_provinces = read_csv(filepath3)

    election_data_2014[0].append('Leader Home Province' )
    for leader in election_data_2014[1:]:
        get_leader_province(home_provinces, leader)

    election_data_2019[0].append('Leader Home Province' )
    for leader in election_data_2019[1:]:
        get_leader_province(home_provinces, leader)

    # PROBLEM 4.2
    print('\nProblem 4:\n')
    kn_seats_2014 = get_seats_by_province(election_data_2014, 'kwazulu-natal')
    gauteng_seats_2019 = get_seats_by_province(election_data_2019, 'GAUTENG')

    
    # PROBLEM 5.2
    print('\nProblem 5:\n')
    for data in election_data_2019[1:]:
        diff = get_seat_difference(election_data_2014, data)
        data.append(diff)
    election_data_2019[0].append('Seat Difference')
    # PROBLEM 6.2
    
    filepath4 = './final_election_data_2019.csv'
    write_csv(filepath4, election_data_2019[1:], headers=election_data_2019[0])
    
# WARN: do not modify or remove the following if statement.

if __name__ == '__main__':
    main()