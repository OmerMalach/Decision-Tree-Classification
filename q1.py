import copy
import numpy as np
import pandas as pd
import random
from pprint import pprint
from scipy import stats
import tqdm

pd.set_option("display.max_columns", None)
global column_titles


def adjusting_data_frame():  # method reads file and fixes heading for later process
    global column_titles
    df = pd.read_csv(f'SeoulBikeData.csv', encoding='unicode_escape')
    df = df.drop('Date', axis=1)
    df = df.rename(columns={'Wind speed (m/s)': 'Wind-Speed'})
    df = df.rename(columns={'Solar Radiation (MJ/m2)': 'Solar-Radiation'})
    df = df.rename(columns={'Dew point temperature(°C)': 'Dew-Point-Temperature(°C)'})
    df = df.rename(columns={'Functioning Day': 'Functioning-Day'})
    df = df.rename(columns={'Snowfall (cm)': 'Snowfall(cm)'})
    df = df.rename(columns={'Visibility (10m)': 'Visibility-(10m)'})
    df = df.rename(columns={'Holiday': 'is-it-Holiday?'})
    df = df.rename(columns={'Seasons': 'Season'})
    df['Holiday'] = df.apply(fix_no_holiday_space, axis=1)
    df['Hour'] = df.apply(bin_hours, axis=1)
    df = df.drop("is-it-Holiday?", axis=1)
    df = df.rename(columns={'Holiday': 'is-it-Holiday?'})
    column_titles = df.columns
    return df


def fix_no_holiday_space(example):  # space caused problems > fixed it
    if example['is-it-Holiday?'] == 'No Holiday':
        return 'No-Holiday'
    else:
        return 'Holiday'


def bin_hours(example):  # binning the hours made much more sense
    if 1 <= int(example['Hour']) < 5:
        return 'late-night'
    elif 5 <= int(example['Hour']) < 12:
        return 'morning'
    elif 12 <= int(example['Hour']) < 15:
        return 'after-noon'
    elif 15 <= int(example['Hour']) < 17:
        return 'late-after-noon'
    elif 17 <= int(example['Hour']) < 20:
        return 'evening'
    else:
        return 'night'


def build_tree(ratio):  # method creates decision tree
    df = adjusting_data_frame()
    train_df, test_df = train_test_split(df, ratio)
    tree = decision_tree_algorithm(train_df)
    pprint(tree[0], width=5, indent=1, compact=True)
    accuracy = calculate_accuracy(test_df, tree[0])
    print('after using ' + str(ratio) + '% of the data for training,')
    print('and ' + str(1 - ratio) + '% for testing...')
    print('the reported error is: ' + str(accuracy[0]))


def arrange_row(record):
    # add features name, label column
    record = record.rename({1: 'Hour', 2: 'Temperature(°C)', 3: 'Humidity(%)', 4: 'Wind-Speed', 5: 'Visibility-(10m)',
                            6: 'Dew-Point-Temperature(°C)', 7: 'Solar-Radiation', 8: 'Rainfall(mm)', 9: 'Snowfall(cm)',
                            10: 'Season', 11: ' Functioning-Day', 12: 'is-it-Holiday?'}, axis=1)
    if record['is-it-Holiday?'] == 'No Holiday':
        record['is-it-Holiday?'] = 'No-Holiday'
    else:
        record['is-it-Holiday?'] = 'Holiday'
    if 1 <= int(record['Hour']) < 5:
        record['Hour'] = 'late-night'
    elif 5 <= int(record['Hour']) < 12:
        record['Hour'] = 'morning'
    elif 12 <= int(record['Hour']) < 15:
        record['Hour'] = 'after-noon'
    elif 15 <= int(record['Hour']) < 17:
        record['Hour'] = 'late-after-noon'
    elif 17 <= int(record['Hour']) < 20:
        record['Hour'] = 'evening'
    else:
        record['Hour'] = 'night'
    return record


def is_busy(row_input):
    row_to_predict = np.array(row_input)
    example = pd.Series(row_to_predict)
    example = example.drop([0])
    df = adjusting_data_frame()
    example = arrange_row(example)  # remove date column (not used)
    train_df, test_df = train_test_split(df, 1)
    tree = decision_tree_algorithm(train_df)
    answer = classify_example(example, tree[0])
    if answer == 'busy':
        print(str(1) + ' ->>(busy)')
    else:
        return print(str(0) + ' ->>(not busy)')


def get_potential_splits(data):  # method creates a dictionary with the key being the column of the attribute
    potential_splits = {}  # and as value an array containing all possible splitting points for that attribute
    _, n_columns = data.shape  # we only care for the columns length
    for column_index in range(1, n_columns):  # excluding the first column which is the "label"
        potential_splits[column_index] = []  # place an array within
        values = data[:, column_index]  # take all rows of the specific attribute
        unique_values = list(np.unique(values))  # removing duplicates
        for index in range(len(unique_values)):
            if is_digit(unique_values[index]) and column_index != 1:  # column_index=1(hours)>> no splitting needed
                if len(unique_values) == 1:
                    potential_splits[column_index] = unique_values
                elif index != 0:  # skipping the first element because it has no previous
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = (current_value + previous_value) / 2  # calculating the value between 2 points
                    potential_splits[column_index].append(potential_split)
            else:
                potential_splits[column_index] = unique_values
                break
    return potential_splits


def split_data(data, split_column, split_value, for_real):  # method designed to split the data into 2 sections
    split_column_values = data[:, split_column]
    if for_real:  # produce split that will be used
        if is_digit(split_value) and is_digit(split_column_values[0]):
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values > split_value]
            return data_below, data_above
    else:  # produce split for testing which is the best split
        if is_digit(split_value) and split_column != 1:
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values > split_value]
        else:
            data_below = data[split_column_values == split_value]
            data_above = data[split_column_values != split_value]
        return data_below, data_above


def split_data_by_category(data, split_column):  # method designed to split the data into 2 sections
    split_column_values = data[:, split_column]
    unique_values = np.unique(split_column_values)  # removing duplicates
    temp1 = data[split_column_values == unique_values[0]]
    data_split = {unique_values[0]: temp1}  # create first element in dic
    for index in range(1, len(unique_values)):
        attribute = "{}".format(unique_values[index])
        data_split[attribute] = data[split_column_values == unique_values[index]]
    return data_split  # returns a dic with the attribute as key and value as the corresponding data


def cross_validation_split(data, folds):  # method divide the data into k folds
    data_folds = list()
    data_size = data.shape
    fold_size = int(data_size[0] / folds)
    for i in range(folds):
        indices = data.index.tolist()
        k_portion = random.sample(population=indices, k=fold_size)
        k_data = data.loc[k_portion]
        k_data.reset_index(drop=True)
        data_folds.append(k_data)
        data = data.drop(k_portion)
        data.reset_index(drop=True)
    return data_folds


def tree_error(k):  # method is responsible of saving all error values (after validating the k fold) and
    df = adjusting_data_frame()  # print the average when process is done
    splits = cross_validation_split(df, k)  # call for splitting the data
    errors = np.array([])
    for index in tqdm.tqdm(range(k), 'Calculating ' + str(k) + ' Fold Cross Validation On Tree .....'
                                                               ' It while take a minute.... ;)'):
        train, test_data = pull_validation(splits, index)
        tree = decision_tree_algorithm(train)
        accuracy = calculate_accuracy(test_data, tree[0])  # calculate the accuracy of the specific testing data
        test_data = accuracy[1]
        splits = fix_after_label(splits, test_data, index)
        errors = np.append(errors, accuracy[0])
    average_error = errors.sum() / k
    print('After preforming ' + str(k) + ' fold cross-validation')
    print('The error mean of the tree is :' + str(average_error))


def fix_after_label(splits, test1, index):
    splits[index] = test1
    return splits


def pull_validation(splits, k):  # method takes out the k subsection and combines all other sections into training set
    test_d = splits[k]
    counter = 0
    train = None
    for i in range(len(splits)):
        if i != k:
            if counter == 0:
                train = splits[i].reset_index(drop=True)
                counter += 1
                continue
            else:
                splits[i] = splits[i].reset_index(drop=True)
                train = pd.concat([train, splits[i]], axis=0)
    return train, test_d


def calculate_entropy(data):
    label_column = data[:, 0]  # get all "label" records from data
    busy_count = (label_column > 650).sum()  # count the number of records above the busy thresh-hold
    if len(label_column) > 0:
        busy_pro = busy_count / len(label_column)  # calculate proportion
    else:
        busy_pro = 0
    not_busy_p = 1 - busy_pro
    if busy_pro != 0 and not_busy_p != 0:
        probabilities = np.array([busy_pro, not_busy_p])
        entropy = sum(probabilities * -np.log2(probabilities))  # element wise
    else:  # we know that if all instances belong the one group there is no randomness >> entropy = 0
        entropy = 0
    return entropy


def calculate_overall_entropy(data_below, data_above):  # method is being used in "determine_best_split"
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n
    overall_entropy = (p_data_below * calculate_entropy(data_below)
                       + p_data_above * calculate_entropy(data_above))  # average
    return overall_entropy


def determine_best_split(data, potential_splits, titles):  # methods finds the best split
    overall_entropy = 100000
    best_split_column = None
    best_split_value = None
    title_dict = titles[0]
    title_map = titles[1]
    for column_index in range(1, 13):  # loop over all the keys
        if title_dict[title_map[column_index]] == 0:  # meant to stop the function from trying to divide the data by-
            pass  # -attributes that have already been asked
        else:
            for value in potential_splits[column_index]:  # loop over all elements in list
                data_below, data_above = split_data(data, split_column=column_index, split_value=value, for_real=False)
                current_overall_entropy = calculate_overall_entropy(data_below, data_above)
                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
    return best_split_column, best_split_value  # method returns> most meaningful attribute + value that result it


def train_test_split(df, train_ratio):  # method is responsible to divide to data into 2 sections: 1.training 2.test
    test_size = round((1 - train_ratio) * len(df))  # get the actual number of records to save as test data
    indices = df.index.tolist()  # make a list of all indices in file
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def check_purity_true(data):  # method return true if all records are above 650 bikes
    list_bike_rented_count = data[:, 0]  # get all amount of rented bike in a list
    counter = (list_bike_rented_count > 650).sum()
    if counter == len(list_bike_rented_count):
        return True
    else:
        return False


def check_purity_false(data):  # method return true if all records are below 650 bikes
    list_bike_rented_count = data[:, 0]  # get all amount of rented bike in a list
    if (list_bike_rented_count <= 650).sum() == len(list_bike_rented_count):
        return True
    else:
        return False


def busy_p(data):  # method return true if all records are below 650 bikes
    list_bike_rented_count = data[:, 0]  # get all amount of rented bike in a list
    busy_count = (list_bike_rented_count > 650).sum()
    busy_proportion = busy_count / len(data)
    return busy_proportion, 1 - busy_proportion


def busy_counter(data):  # method return true if all records are below 650 bikes
    list_bike_rented_count = data[:, 0]  # get all amount of rented bike in a list
    busy_count = (list_bike_rented_count > 650).sum()
    return busy_count, len(data) - busy_count


def classify_data(data):  # method is being called after the check for purity has been done, meaning either the data is
    label_column = data[:, 0]  # entirely "busy" or entirely not busy
    busy_boolean = label_column > 650
    number_of_busy = len(label_column[busy_boolean])
    not_busy_boolean = label_column <= 650
    number_of_not = len(label_column[not_busy_boolean])
    if number_of_busy > number_of_not:
        classification = 'busy'
    else:
        classification = 'not-busy'
    return classification


def calculate_chai(proportion_tup, data):  # method is responsible of calculating chai score
    expected_p = proportion_tup[0] * len(data)
    expected_n = proportion_tup[1] * len(data)
    actual_p, actual_n = busy_counter(data)
    chai_score = (((actual_p - expected_p) ** 2) / expected_p) + (((actual_n - expected_n) ** 2) / expected_n)
    return chai_score


def chai_test(chai_total, data):
    total_deviation = chai_total.sum()
    degree_of_freedom = len(data) - 1
    critical = stats.chi2.ppf(0.95, degree_of_freedom)
    if total_deviation < critical:
        return True  # meaning we wont reject h0 > question has meaning.
    else:
        return False


def set_title_dic(df):  # method create 2 dict that help us later on know which attribute have already been exhausted
    titles = df.columns
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    b = titles
    dict1 = {}
    dict2 = {}
    for A, B in zip(a, b):
        dict1[B] = A
    for A, B in zip(a, b):
        dict2[A] = B
    return dict1, dict2


def no_more_attribute(titles):  # method recognizes when all attribute have been exhausted
    answer = True
    for column_index in range(len(titles[0])):
        if titles[0][titles[1][column_index]] != 0:
            answer = False
    return answer


def decision_tree_algorithm(df, counter=0, min_samples=42, max_depth=420, proportion_tup=None, title_dict=None):
    desired_col = 0
    if counter == 0:  # method generates  decision tree
        data = df.values  # data preparations is needed first time this function is called
        titles = set_title_dic(df)
    else:
        titles = title_dict
        data = df
    if counter != 0 and len(data) == 0:  # base case (examples is empty)
        return None
    elif check_purity_true(data) or check_purity_false(data) or (len(data) < min_samples) or (counter == max_depth) \
            or no_more_attribute(titles):  # base cases
        chai_score = calculate_chai(proportion_tup, data)
        classification = classify_data(data)
        return classification, chai_score
    else:  # recursive part
        counter += 1
        potential_splits = get_potential_splits(data)  # helpers
        split_column, split_value = determine_best_split(data, potential_splits, titles)
        attribute = None
        for index, name in titles[1].items():  # making sure were getting the right attribute
            if index == split_column:
                attribute = name
                break
        if attribute != 'Season' and attribute != 'Hour' and attribute != 'Functioning-Day' \
                and attribute != 'is-it-Holiday?':
            question1 = '{} <= {}'.format(attribute, split_value)  # instantiate sub-tree for continues attribute
            categorical = False
        else:
            question1 = '{} - ?'.format(attribute)  # instantiate sub-tree for categorical
            categorical = True
        sub_trees = {question1: []}  # tree in the form of a dictionary
        proportion_tup = busy_p(data)
        chai_total = np.array([])
        stop_search = False
        for index1, value_title in titles[1].items():
            for key_title, index2 in titles[0].items():
                if value_title == key_title == attribute:
                    desired_col = index1  # making sure were getting the right column
                    stop_search = True
                    break
            if stop_search:
                break
        if categorical:
            title_map = titles[1]
            title_update1 = fix_title_index(titles, split_column)  # removing the about to get split column
            titles = title_update1, title_map  # from becoming an option for split
            dp_titles = copy.deepcopy(titles)
            splits = split_data_by_category(data, desired_col)
            for val in (potential_splits[desired_col]):
                question2 = '{} is {}'.format(attribute, val)  # in case of categorical there are 2 questions
                sub_tree = {question2: []}  # tree in the form of a dictionary
                classification_tup = decision_tree_algorithm(splits[val], counter, min_samples, max_depth,
                                                             proportion_tup, dp_titles)  # find below(recursion)
                if classification_tup is None:  # meaning the "son"s len(data) (examples) was 0
                    chai_score = calculate_chai(proportion_tup, data)
                    classification = classify_data(data)  # base cases
                    return classification, chai_score
                else:
                    answer, chai_score = classification_tup
                    chai_total = np.append(chai_total, chai_score)
                    sub_tree[question2].append(answer)
                    sub_trees[question1].append(sub_tree)
        else:
            data_below, data_above = split_data(data, desired_col, split_value, True)  # same but simpler (continues)
            title_map = titles[1]
            title_update2 = fix_title_index(titles, split_column)
            titles = title_update2, title_map
            dp_titles_yes = copy.deepcopy(titles)
            dp_titles_no = copy.deepcopy(titles)
            classification_tup_yes = decision_tree_algorithm(data_below, counter, min_samples, max_depth,
                                                             proportion_tup, dp_titles_yes)  # find below(recursion)
            classification_tup_no = decision_tree_algorithm(data_above, counter, min_samples, max_depth, proportion_tup,
                                                            dp_titles_no)  # find above(recursion)
            if (classification_tup_no is None) or (classification_tup_yes is None):
                chai_score = calculate_chai(proportion_tup, data)
                classification = classify_data(data)  # base cases
                return classification, chai_score
            else:
                chai_total = np.append(chai_total, classification_tup_yes[1])
                chai_total = np.append(chai_total, classification_tup_no[1])
                sub_trees[question1].append(classification_tup_yes[0])  # add sub tree into main tree
                sub_trees[question1].append(classification_tup_no[0])
    if chai_test(chai_total, data):  # chai test pruning
        chai_score = calculate_chai(proportion_tup, data)
        return sub_trees, chai_score
    else:
        chai_score = calculate_chai(proportion_tup, data)  # in case chai test returned false > pruning takes place
        classification = classify_data(data)
        return classification, chai_score  # pruning


def fix_title_index(titles, split_column):  # method keeps the record of which attribute has been split
    desired_col_name = titles[1][split_column]  # so that attribute wont be an option for the "sons"
    down_garde_by = 0
    fix = False
    for name, index in titles[0].items():
        below = index - 1
        if name == desired_col_name:
            titles[0][name] = 0
            down_garde_by = index - below
            fix = True
        if fix and titles[0][name] != 0:
            new_index = index - down_garde_by
            titles[0][name] = new_index
    return titles[0]


def is_digit(n):  # method being used in classify_example
    try:  # returns true if a string ia actually a number
        float(n)
        return True
    except ValueError:
        return False


def calculate_accuracy(df, tree):  # reports error rate
    df['classification'] = df.apply(classify_example, axis=1, args=(tree,))  # args=(tree,) addition for function
    df['label'] = df.apply(generate_label_column, axis=1)
    df['classification_correct'] = df['classification'] == df['label']  # creating a third column for prediction
    accuracy = df['classification_correct'].mean()
    error = 1 - accuracy
    df = df.drop('classification', axis=1)
    df = df.drop('label', axis=1)
    df = df.drop('classification_correct', axis=1)
    return error, df


def generate_label_column(example):  # helps with accuracy calculations
    if example['Rented Bike Count'] > 650:
        return 'busy'
    else:
        return 'not-busy'


def classify_example(example, tree):  # works for single record
    answer = None
    busy_list = ['busy']
    not_list = ['not-busy']
    if isinstance(tree, str):
        return tree
    else:
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split()
        if comparison_operator == '<=':
            try:
                if example[feature_name] <= float(value):  # ask question
                    answer = tree[question][0]
                else:
                    answer = tree[question][1]
            except:
                TypeError
                if float(example[feature_name]) <= float(value):
                    answer = tree[question][0]
                else:
                    answer = tree[question][1]
        else:
            if tree[question] == busy_list or tree[question] == not_list:
                if tree[question] == busy_list:
                    return 'busy'
                else:
                    return 'not-busy'
            else:
                for i in range(len(tree[question])):
                    question2 = list(tree[question][i])[0]
                    feature_name, comparison_operator, value = question2.split()
                    if example[feature_name] == value:  # ask question
                        answer = tree[question][i]
                        break
                if answer is None:
                    residual_tree = tree[question][0]
                    answer = classify_example(example, residual_tree)  # 2 questions for categorical >>
    if not isinstance(answer, dict):  # base case                                 # just a bit of recursion ;)
        return answer
    else:  # recursive part
        residual_tree = answer
        return classify_example(example, residual_tree)


if __name__ == '__main__':
    print('Run Directly')
    # build_tree(0.6)
    # tree_error(10)
    # test_n = ['01/12/2017', 5, -6.4, 37, 1.5, 2000, -18.7, 0, 0, 0, 'Winter', 'No Holiday', 'Yes']
    # test_p = ['12/07/2018', 21, 26.3, 81, 0.8, 1910, 22.7, 0, 0, 0, 'Summer', 'No Holiday', 'Yes']
    # is_busy(test_p)
    # is_busy(test_n)
else:
    print("Run From Import")
