import pandas as pd
import matplotlib.pyplot as plt


class Super_Normalization():
    # Declaring static url of data
    url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    # Declaring standard colors for all countries. These colors will be fixed across all plots
    COLOR = {'France': '#44a2c7', 'Germany': '#f42fa2', 'Finland': '#3b2b99', 'Russia': '#f41f09', 'United Kingdom': '#a25ee1', 'Italy': '#91190a', 'Spain': '#0258bb', 'Sweden': '#2f65bf', 'Slovenia': '#d2ba0d', 'Denmark': '#a83e40', 'Estonia': '#8d3b08', 'Belgium': '#9dbeaa', 'Greece': '#5e4b98', 'Luxembourg': '#b51c57', 'Norway': '#1e5c3e', 'Switzerland': '#2106f2', 'Albania': '#53ace0', 'Austria': '#223406', 'Croatia': '#4f5026', 'Latvia': '#c923cc', 'Romania': '#ae7a50', 'North Macedonia': '#59a61f', 'Serbia': '#96510b', 'Netherlands': '#0525ba', 'Belarus': '#bc6309', 'Iceland': '#e2d4b3',
             'Monaco': '#c28cc5', 'Ireland': '#62b93e', 'San Marino': '#668d02', 'Czechia': '#bcaab2', 'Portugal': '#4851de', 'Andorra': '#c6d06f', 'Ukraine': '#5a7c84', 'Hungary': '#1d9d53', 'Liechtenstein': '#b3a9c0', 'Faeroe Islands': '#d2eef6', 'Poland': '#a55d32', 'Gibraltar': '#bfa438', 'Bosnia and Herzegovina': '#e3cad7', 'Malta': '#2346a6', 'Slovakia': '#732bc8', 'Vatican': '#e52e53', 'Moldova': '#998396', 'Cyprus': '#400089', 'Bulgaria': '#fbe7a8', 'Kosovo': '#f2c023', 'Montenegro': '#a2c1bd', 'Lithuania': '#18a424', 'Isle of Man': '#4a1793', 'Guernsey': '#171d71', 'Jersey': '#256586'}

    def __init__(self):
        self.data_frame = pd.read_csv(Super_Normalization.url)
        self.group_by_country = ()

    def Filter_Column(self):
        filt = self.data_frame.continent == "Europe"
        self.data_frame = self.data_frame.loc[filt]
        self.data_frame.set_index("date", inplace=True)
        self.data_frame.rename(columns={"location": "country"}, inplace=True)
        self.data_frame = self.data_frame[[
            "country", "new_cases", "new_deaths", "hosp_patients", "icu_patients"]]

    def Grouping_by_country(self):
        self.group_by_country = self.data_frame.groupby("country")

    # storing the list of countries with whole column as empty values for a specific parameter

    def list_of_Nan_countries(self, parameter):

        array_of_null_countries = []

        for grouped_country_name, grouped_country_database in self.group_by_country:
            if (len(grouped_country_database) == grouped_country_database[parameter].isnull().sum()):
                array_of_null_countries.append(grouped_country_name)
        return array_of_null_countries

    # dataframe of countries for a particular parameter in the form of Pivot Table

    def get_country_df_for_particular_parameter(self, parameter):
        country_df_particular_parameter = self.data_frame.pivot(
            index="date", columns="country", values=parameter)
        return country_df_particular_parameter

    # remove the columns of the list of Nan countries from the dataframe

    def delete_Nan_countries_from_df(self, array_of_null_countries, country_df_particular_parameter):

        for df_country in country_df_particular_parameter.columns:
            for array_country in array_of_null_countries:
                if array_country in df_country:
                    del country_df_particular_parameter[df_country]
        return country_df_particular_parameter

    # storing the name of the parameters from the data frame of a list
    def getparameter_array(self):
        col_array = []
        for col_name in self.data_frame.columns:
            if(col_name != "date" and col_name != "country"):
                col_array.append(col_name)
        return col_array

    # returns the final clean dataset for each parameter in the form of dictionary
    # where the parameter are the keys and data frames are the values
    def get_final_df_Dictionary(self, rolling_days = 14):
        self.Filter_Column()
        self.data_frame.reset_index(inplace=True)
        self.Grouping_by_country()
        dictionary = {}
        col_array = self.getparameter_array()
        for parameter in col_array:
            Nan_ans = self.list_of_Nan_countries(parameter)
            df_ans = self.get_country_df_for_particular_parameter(parameter)
            deleted_nan_country_df = self.delete_Nan_countries_from_df(
                Nan_ans, df_ans)
            # final missing values "inside" the dataframe are filled using linear interpolation method
            interpolated_df = deleted_nan_country_df.interpolate(
                limit_area="inside")
            #finding the rolling _average for better visualisation
            rolling_avg_df = rolling_average(interpolated_df, rolling_days)
            dictionary[parameter] = rolling_avg_df

        return dictionary

    # The Type argument is the type of Normalization
    # The catagory argument is a catagory such as 'new_cases'
    # The countries = None arguments plots all countries if there are no specified countries
    def plot_data_frame(self, DataFrame, Type, catagory, countries=None):
        DataFrame.reset_index(inplace=True)
        DataFrame['date'] = pd.to_datetime(DataFrame['date'])

        if countries != None:
            for column in countries:
                plt.plot(
                    DataFrame.date, DataFrame[column], color=Super_Normalization.COLOR[column], label=column)
        if countries == None:
            for column in DataFrame:
                if column == 'date':
                    continue
                plt.plot(
                    DataFrame['date'], DataFrame[column], color=Super_Normalization.COLOR[column], label=column)
        plt.title("Normalizing each country with " +
                  Type + " Maximum " + catagory)
        plt.xlabel("Dates")
        plt.ylabel("Normalized to 1")
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.show()
        
    
    # new_df = dataframe with date as index and countries as column
    # rolling_days = number of days you want to take average of (strongly recommended that the value should be a multiple of 7)
    def rolling_average(self, new_df, rolling_days):
        
        row_count = new_df.shape[0]
        column_count = new_df.shape[1]

        temp_data_frame = new_df.copy()

        for country_index in range(column_count):

            index_counter = 0

            # for each date in each country
            for date_index in range(row_count):

                # surpass all the Nan values in the dataframe then only proceeding
                if not (np.isnan(new_df.iloc[date_index, country_index])):

                    # summing of the rolling days from the copied data frame
                    # assuming that the dataframe is larger than the rolling_days parameter
                    rolling_counter_index = date_index
                    ending_rolling_counter_index = date_index + rolling_days 

                    # finding the mean of all the days within the rolling days window
                    rolling_days_mean = temp_data_frame.iloc[rolling_counter_index:
                                                             ending_rolling_counter_index, country_index].mean()       
                    
                    # updating the new value in the original dataframe
                    new_df.iloc[rolling_counter_index,country_index] = rolling_days_mean

                    # reached the end of the dataframe
                    if(rolling_counter_index == row_count-1):
                        break

                    # this function will work for the first number of rolling days
                    if index_counter < rolling_days:
                        # removing (here filling with Nan value) the first rolling days values from the dataframe
                        new_df.iloc[date_index, country_index] = np.nan
                        index_counter += 1
        return new_df
    
    ###########################Beta function
    ###########################Conversion of the dataframe into three states
    def num_to_sign_converter(self, val):
    
    if val <= 0.333:
        #assign the value - (mild)
        return "-"
        
    elif val > 0.33 and val <= 0.666:
         #assign the value + (moderate)
        return "+"
    
    elif val > 0.666 and val <= 1:
         #assign the value ++ (severe)
        return "++"
    else:
        return np.nan
    
    def save_and_convert_to_three_states(self, dataframe, file_name):
        final_df = dataframe.applymap(lambda x : self.num_to_sign_converter(x))
        final_df.to_excel(file_name + ".xlsx")
        
    
        
        
        

# Sp = Super_Normalization()
# #you are access the dataframe for each the four parameter by just using passing parameter as the key value
# #in the dictionary
# dict = Sp.get_final_df_Dictionary()
# import pandas as pd
=======
import Segregating_countries_by_population_and_density as SCPD
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Super_Normalization():
    # Declaring static url of data
    url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
    # Declaring standard colors for all countries. These colors will be fixed across all plots
    COLOR = {'France': '#44a2c7', 'Germany': '#f42fa2', 'Finland': '#3b2b99', 'Russia': '#f41f09', 'United Kingdom': '#a25ee1', 'Italy': '#91190a', 'Spain': '#0258bb', 'Sweden': '#2f65bf', 'Slovenia': '#d2ba0d', 'Denmark': '#a83e40', 'Estonia': '#8d3b08', 'Belgium': '#9dbeaa', 'Greece': '#5e4b98', 'Luxembourg': '#b51c57', 'Norway': '#1e5c3e', 'Switzerland': '#2106f2', 'Albania': '#53ace0', 'Austria': '#223406', 'Croatia': '#4f5026', 'Latvia': '#c923cc', 'Romania': '#ae7a50', 'North Macedonia': '#59a61f', 'Serbia': '#96510b', 'Netherlands': '#0525ba', 'Belarus': '#bc6309', 'Iceland': '#e2d4b3',
             'Monaco': '#c28cc5', 'Ireland': '#62b93e', 'San Marino': '#668d02', 'Czechia': '#bcaab2', 'Portugal': '#4851de', 'Andorra': '#c6d06f', 'Ukraine': '#5a7c84', 'Hungary': '#1d9d53', 'Liechtenstein': '#b3a9c0', 'Faeroe Islands': '#d2eef6', 'Poland': '#a55d32', 'Gibraltar': '#bfa438', 'Bosnia and Herzegovina': '#e3cad7', 'Malta': '#2346a6', 'Slovakia': '#732bc8', 'Vatican': '#e52e53', 'Moldova': '#998396', 'Cyprus': '#400089', 'Bulgaria': '#fbe7a8', 'Kosovo': '#f2c023', 'Montenegro': '#a2c1bd', 'Lithuania': '#18a424', 'Isle of Man': '#4a1793', 'Guernsey': '#171d71', 'Jersey': '#256586'}
    # All countries:
    All_countries = ['Andorra', 'Austria', 'Denmark', 'Estonia', 'Faeroe Islands', 'France', 'Iceland',
                     'Isle of Man', 'Latvia', 'Liechtenstein', 'Lithuania', 'Netherlands', 'Portugal', 'San Marino', 'Slovenia', 'Belgium', 'Croatia', 'Cyprus', 'Czechia', 'Finland', 'Germany', 'Gibraltar', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Monaco', 'Montenegro', 'Norway', 'Romania', 'Serbia', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'Albania', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Moldova', 'North Macedonia', 'Poland', 'Russia', 'Ukraine', 'Vatican']
    
    # Countries seperated by density
    density_in_ascending_order = SCPD.Density_in_order
    

    def __init__(self):
        self.data_frame = pd.read_csv(Super_Normalization.url)
        self.group_by_country = ()
        # Dataframe containing data about population and population density of european countries
        self.population_and_density_df = pd.read_csv(
            'Population_and_population_density.csv')

    # Setting index for population density dataframe
    def reorient_pop_and_pop_density(self):
        self.population_and_density_df.rename(
            columns={'name': 'Country'}, inplace=True)
        
        #correcting the country name in the Population_and_population_density.csv file
        self.population_and_density_df.iloc[11, 0] = "Czechia"
        self.population_and_density_df.iloc[42, 0] = "Faeroe Islands"
        self.population_and_density_df.set_index('Country', inplace=True)

    def Filter_Column(self):
        filt = self.data_frame.continent == "Europe"
        self.data_frame = self.data_frame.loc[filt]
        self.data_frame.set_index("date", inplace=True)
        self.data_frame.rename(columns={"location": "country"}, inplace=True)
        self.data_frame = self.data_frame[[
            "country", "new_cases", "new_deaths", "hosp_patients", "icu_patients"]]

    def Grouping_by_country(self):
        self.group_by_country = self.data_frame.groupby("country")

    # storing the list of countries with whole column as empty values for a specific parameter
    def list_of_Nan_countries(self, parameter):

        array_of_null_countries = []

        for grouped_country_name, grouped_country_database in self.group_by_country:
            if (len(grouped_country_database) == grouped_country_database[parameter].isnull().sum()):
                array_of_null_countries.append(grouped_country_name)
        return array_of_null_countries

    # dataframe of countries for a particular parameter in the form of Pivot Table
    def get_country_df_for_particular_parameter(self, parameter):
        country_df_particular_parameter = self.data_frame.pivot(
            index="date", columns="country", values=parameter)
        return country_df_particular_parameter

    # remove the columns of the list of Nan countries from the dataframe
    def delete_Nan_countries_from_df(self, array_of_null_countries, country_df_particular_parameter):

        for df_country in country_df_particular_parameter.columns:
            for array_country in array_of_null_countries:
                if array_country in df_country:
                    del country_df_particular_parameter[df_country]
        return country_df_particular_parameter

    # storing the name of the parameters from the data frame of a list
    def getparameter_array(self):
        col_array = []
        for col_name in self.data_frame.columns:
            if(col_name != "date" and col_name != "country"):
                col_array.append(col_name)
        return col_array

    # Dividng each country's data by its population
    def divide_by_population(self, DataFrame):
        for country in DataFrame.columns:
            if country in self.population_and_density_df.index:
                DataFrame[country] = DataFrame[country].apply(
                    lambda x: x/self.population_and_density_df.loc[country, "pop"])
        return DataFrame

    # returns the final clean dataset for each parameter in the form of dictionary
    # where the parameter are the keys and data frames are the values

    def get_final_df_Dictionary(self, rolling_days=14):
        self.reorient_pop_and_pop_density()
        self.Filter_Column()
        self.data_frame.reset_index(inplace=True)
        self.Grouping_by_country()
        dictionary = {}
        col_array = self.getparameter_array()
        for parameter in col_array:
            Nan_ans = self.list_of_Nan_countries(parameter)
            df_ans = self.get_country_df_for_particular_parameter(parameter)
            deleted_nan_country_df = self.delete_Nan_countries_from_df(
                Nan_ans, df_ans)

            # final missing values "inside" the dataframe are filled using linear interpolation method

            interpolated_df = deleted_nan_country_df.interpolate(
                limit_area="inside")
            for country in interpolated_df.columns:
                if country not in self.population_and_density_df.index:
                    interpolated_df = interpolated_df.drop([country], axis=1)

            interpolated_df = self.divide_by_population(
                DataFrame=interpolated_df)
            interpolated_df = self.rolling_average(
                interpolated_df, rolling_days)
            dictionary[parameter] = interpolated_df

        return dictionary

    # Segrgates Countries based on their maximum in graphs
    @staticmethod
    def segregate_countries(DataFrame_to_be_segregated, first_threshold=0.3, second_threshold=0.6, third_threshold=1.1):
        Dataframe = DataFrame_to_be_segregated.copy()
        c1 = []
        c2 = []
        c3 = []
        for column in Dataframe.columns:
            Max = Dataframe[column].max()
            if Max < first_threshold:
                c1.append(str(column))
                continue
            if Max < second_threshold:
                c2.append(str(column))
                continue
            if Max <= third_threshold:
                c3.append(str(column))
        country_groups = [c1, c2, c3]
        return country_groups

    # Conversion of the dataframe into three states
    @staticmethod
    def num_to_sign_converter(val, Max=1):

        if val <= Max/3:
            # assign the value - (mild)
            return "-"

        elif val > Max/3 and val <= (2/3)*Max:
            # assign the value + (moderate)
            return "+"

        elif Max/val > (2/3)*Max and val <= Max:
            # assign the value ++ (severe)
            return "++"
        else:
            return np.nan
    # Saving Dataframe as excel file
    # Converting DataFrame to Three_states

    @staticmethod
    def save_and_convert_to_three_states_with_max_1(dataframe, file_name):
        Three_states_df = dataframe.applymap(
            lambda x: Super_Normalization.num_to_sign_converter(x))
        print(Three_states_df)
        Three_states_df.to_excel(file_name + ".xlsx")

    @staticmethod
    def save_and_convert_to_three_state_with_country_specific_max(dataframe, file_name):
        Three_states_df = pd.DataFrame()
        for country in dataframe.columns:
            max_of_country = dataframe[country].max()
            Three_states_df[country] = dataframe[country].apply(
                lambda x: Super_Normalization.num_to_sign_converter(x, max_of_country))
        Three_states_df.to_excel(file_name + ".xlsx")

    @staticmethod
    def assign_transition(state_a, state_b, Stochastic_matrix):
        if state_a == '-' and state_b == '-':
            Stochastic_matrix[0][0] += 1
        if state_a == '-' and state_b == '+':
            Stochastic_matrix[0][1] += 1
        if state_a == '-' and state_b == '++':
            Stochastic_matrix[0][2] += 1
        if state_a == '+' and state_b == '-':
            Stochastic_matrix[1][0] += 1
        if state_a == '+' and state_b == '+':
            Stochastic_matrix[1][1] += 1
        if state_a == '+' and state_b == '++':
            Stochastic_matrix[1][2] += 1
        if state_a == '++' and state_b == '-':
            Stochastic_matrix[2][0] += 1
        if state_a == '++' and state_b == '+':
            Stochastic_matrix[2][1] += 1
        if state_a == '++' and state_b == '++':
            Stochastic_matrix[2][2] += 1

    @staticmethod
    def get_transition_matrix(dataframe=pd.DataFrame(), Country="Germany"):
        for j in range(0, dataframe.shape[1]):
            if dataframe.columns[j] == Country:
                column_index = j
                break
        Numb_rows = dataframe.shape[0]
        Stochastic_Matrix = [[0]*3]*3
        Stochastic_Matrix_np = np.array(Stochastic_Matrix)
        for i in range(0, Numb_rows-1):
            Super_Normalization.assign_transition(
                state_a=dataframe.iloc[i, column_index], state_b=dataframe.iloc[i+1, column_index], Stochastic_matrix=Stochastic_Matrix_np)
        print(Stochastic_Matrix_np)
        sum_of_rows = np.sum(Stochastic_Matrix_np, 1)
        Stochastic_Matrix_np = Stochastic_Matrix_np.astype(np.float16)
        print(sum_of_rows)
        for m in range(3):
            for n in range(3):
                if sum_of_rows[m] == 0:
                    print("yes")
                    continue
                Stochastic_Matrix_np[m][n] = Stochastic_Matrix_np[m][n] / \
                    sum_of_rows[m]
        return Stochastic_Matrix_np
    # The Type argument is the type of Normalization
    # path_to_save give the location in which the plot is saved
    # name on saving gives the name of the figure as well as the name of the file
    # The countries = None arguments plots all countries if there are no specified countries

    def plot_data_frame(self, DathFrame_to_be_plotted, countries=[], path_to_save="", name_on_saving=""):
        DataFrame = DathFrame_to_be_plotted.copy(deep=True)
        # finding the rolling _average for better visualisation
        DataFrame.reset_index(inplace=True)
        DataFrame['date'] = pd.to_datetime(DataFrame['date'])
        # Making the figure size larger
        plt.figure(figsize=(15, 15))
        if len(countries) != 0:
            for column in countries:
                if column not in DataFrame.columns:
                    continue
                plt.plot(
                    DataFrame.date, DataFrame[column], color=Super_Normalization.COLOR[column], label=column)
                # plt.xticks(DataFrame.date[::100])
        if len(countries) == 0:
            for column in DataFrame.columns:
                if column == 'date':
                    continue
                plt.plot(
                    DataFrame['date'], DataFrame[column], color=Super_Normalization.COLOR[column], label=column)
                # plt.xticks(DataFrame.date[::100])
        plt.title(name_on_saving)
        plt.xlabel("Dates")
        plt.ylabel("Normalized to 1")
        plt.xticks(DataFrame.date[::200])
        plt.tick_params(axis='x', labelrotation=0)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
        # plt.savefig(Type + "Static" + "Maximum" + Catagory[i] +'.png', dpi = 300)
        # plt.savefig(path_to_save + name_on_saving)
        plt.show()

    # new_df = dataframe with date as index and countries as column
    # rolling_days = number of days you want to take average of (strongly recommended that the value should be a multiple of 7)

    def rolling_average(self, new_df, rolling_days):

        row_count = new_df.shape[0]
        column_count = new_df.shape[1]

        temp_data_frame = new_df.copy()

        for country_index in range(column_count):

            index_counter = 0

            # for each date in each country
            for date_index in range(row_count):

                # surpass all the Nan values in the dataframe then only proceeding
                if not (np.isnan(new_df.iloc[date_index, country_index])):

                    # summing of the rolling days from the copied data frame
                    # assuming that the dataframe is larger than the rolling_days parameter
                    rolling_counter_index = date_index
                    ending_rolling_counter_index = date_index + rolling_days

                    # finding the mean of all the days within the rolling days window
                    rolling_days_mean = temp_data_frame.iloc[rolling_counter_index:
                                                             ending_rolling_counter_index, country_index].mean()

                    # updating the new value in the original dataframe
                    # doing -1 to put the value into the last element of rolling day window (if rolling_days = 7 then into the 7th element i.e (6th index not 7th index)
                    new_df.iloc[ending_rolling_counter_index -
                                1, country_index] = rolling_days_mean

                    # reached the end of the dataframe
                    if(ending_rolling_counter_index - 1 == row_count-1):
                        break

                    # this function will work for the first number of rolling days
                    # except the last day where we are actually filling the new average value
                    if index_counter < rolling_days - 1:
                        # removing (here filling with Nan value) the first rolling days values from the dataframe
                        new_df.iloc[date_index, country_index] = np.nan
                        index_counter += 1
        return new_df


# Sp = Super_Normalization()
# print(Sp.population_and_density_df)
# #you are access the dataframe for each the four parameter by just using passing parameter as the key value
# #in the dictionary
# dict = Sp.get_final_df_Dictionary()
# import pandas as pd
# Sp.reorient_pop_and_pop_density()
# print(Sp.population_and_density_df.loc["Albania", "pop"])

