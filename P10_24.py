## An appointment class for appointments.
class Appointment():
    # Class variable is created since all objects of Appointment superclass needs to communicate
    # with each other.
    # Inputs: 'description' is the description of the appointment, and 'date' is the date of the
    # appointment. The date format is 'dd-mm-yyyy'
    all_appointments = dict()
    # occursOn checks if the appointment occurs on that day.
    def occursOn(self, year: str, month: str, day: str):
        appointments_for_the_method = list()
        try:
            # Sees if it's a one time appointment.
            for i in Appointment.all_appointments['onetime_appointments']:
                if i[1] == f"{day}-{month}-{year}":
                    appointments_for_the_method.append(i[0])
                          
            # Sees if there are any monthly appointments.
        except:
            pass
        # Checking to see for monthly appointments.
        try:
            for i in Appointment.all_appointments['monthly_appointments']:
                if i[1] == str(day):
                    appointments_for_the_method.append(i[0])
        except:
            pass
        # See if there are any daily appointments.
        try:
            for daily_appointment in Appointment.all_appointments['daily_appointments']:
                appointments_for_the_method.append(daily_appointment)

        except:
            pass
        # If the length of the appintments instance variable is 0, then no appointments are for that
        # specific date.
        if len(appointments_for_the_method) == 0:
            return print(f"There are no appointments for {day}-{month}-{year}, yet.")

        else:
            print(f"All the appointments for {day}-{month}-{year} is:")
            for appointment in appointments_for_the_method:
                print(appointment)

    ## Instance method save, exports the appointments to a csv file.
    # Inputs are: the file name the user wants to save it as.
    # Dependencies: csv package.
    def save(self, file_name):
        import csv
        file = csv.writer(open(f"{file_name}.csv", 'w'))
        for key, val in Appointment.all_appointments.items():
            file.writerow([key, val])


## The class Onetime represents a one-time appointment.
# This class takes in two inputs, 'description' is the description of the appointment, whereas
# 'date' takes in a date in the format: dd-mm-yyyy.
class Onetime(Appointment):
    def __init__(self, description, date):
        if 'onetime_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['onetime_appointments'].append([description, date])
        else:
            Appointment.all_appointments['onetime_appointments'] = [[description, date]]


## The Monthly class is appointments that occurs one time each month.
class Monthly(Appointment):
    # Inputs are a description of the appointment, and what day of the month it occurs.
    def __init__(self, description, day_of_month):
        # The data structure in this subclass is a dictionary that is added to the class variable, where
        # the key is monthly_appointments and the value is a list containing both the description the day.

        # Checking to see if there are any prior monthly appointments in the object.
        if 'monthly_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['monthly_appointments'].append([description, day_of_month])
        # If this is the first monthly appointment, a key with a list is created and inside the list is the description
        # of the appointment.
        else:
            Appointment.all_appointments['monthly_appointments'] = [[description, day_of_month]]


## The Daily class is for daily appointments.
class Daily(Appointment):
    def __init__(self, description):
        # Checking if there's already been created a daily appointment key
        if 'daily_appointments' in Appointment.all_appointments:
            Appointment.all_appointments['daily_appointments'].append(description)
        # If not, a key with a list is created and inside the list is the description
        # of the appointment.
        else:
            Appointment.all_appointments['daily_appointments'] = [description]


# Instantiating an object from Appointment class, not necessary, but easier for the readability
# of the code.
my_appointments = Appointment()

# Adding several appointments as a part of the prompt.
dmv_appo = Onetime('Appointment with the Norwegian DMV', '12-02-2022')
exercise_appo = Daily('Walk my dog.')
eat_appo = Daily('I have to eat dinner')
hom_deadlines = Monthly('Homework deadline', '20')
exam_day = Onetime('I have an exam today', '12-02-2022')

# Checking to see if there are any appointments occuring 12th December, 2022.
#print(my_appointments.occursOn('2022', '12', '12'))


## A class that loads the dataset from the Appointment.save() method, and returns the data.
# Inputs: specify the 'dataset' and the 'type' of the appointment the class should return. This could
# be daily, monthly, or one time appointment types. It has to be the same input as it's stored in. 
# This means that the user has to copy paste one value of the first column for the correct input.
# Dependencies: Pandas, and NumPy.
class AppointmentLoader():
    # Assuming 'type' means the occurence of the appointment-- whether it's a daily appointment,
    # monthly or one-time.
        def __init__(self, dataset, type):
            import pandas as pd
            df = pd.read_csv(dataset, header=None, index_col=0)
            self._type_of_data = type
            self._type_of_appointment = df.loc[type].values
            
            import pandas as pd
            df = pd.read_csv(dataset, header=None, index_col=0)
            type_of_appointment = df.loc[type].values
            self._appointment_list = type_of_appointment[0].strip('][').split(', ')

        # Loading the part of the data that was specified in the 'type' input in the constructor of this class.
        def load(self):
            print("Here's the appointment from the dataset.")
            # The appointments have different data structure, so they have to be treated differently.
            # Checking if the type is daily_appointments.
            if self._type_of_data == 'daily_appointments':
                for count in range(len(self._appointment_list)):
                    print(self._appointment_list[count])
            
            # Checking if the specified type is one time appointments.
            elif self._type_of_data == 'onetime_appointments':
                for count in range(len(self._appointment_list)):
                    if count % 2 == 0:
                        print(f"{self._appointment_list[count].strip(']')}, that occurs on: {self._appointment_list[count +1].strip('[')}")
            
            # Checking if the appointment is monthly.
            elif self._type_of_data == 'monthly_appointments':
                for count in range(len(self._appointment_list)):
                    if count % 2 == 0:
                        print(f"{self._appointment_list[count].strip(']')}, that occurs each: {self._appointment_list[count +1].strip('[')}")

                        
# Saving a csv file as my_appointments.csv from a random object created from previous question.
exam_day.save('my_appointments')

# Creating an instance of AppointmentLoader class, and specifying the type I want the class to focus on.
my_daily_appointments = AppointmentLoader('my_appointments.csv', 'daily_appointments')

# Returning the data from the class.
my_daily_appointments.load()