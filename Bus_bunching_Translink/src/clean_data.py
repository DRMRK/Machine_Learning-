# This file takes original data, cleans it and saves it.
# The rational behind cleaning steps is explained in the notebooks.

from helper.structured import *

# get the original data
ORIGINAL_DATA = os.environ.get("ORIGINAL_DATA")
# save the cleaned data here
CLEANED_DATA = os.environ.get("CLEANED_DATA")

if __name__ == "__main__":
    df = pd.read_pickle(ORIGINAL_DATA)
    # Change date to datetime format
    df['OperationDate'] = pd.to_datetime(df['OperationDate'])
    # Use a helper function to get extra columns from date column
    add_datepart(df, 'OperationDate')
    # Remove Nan values from the data
    df = df.dropna()
    # Drop the DEW column
    df.Dew.dropna()
    # Drop unnecessary columns.
    df = df.drop(['Line', 'Trip', 'TimingPoint', 'MinStopNo', 'MaxStopNo',
                  'ScheduledArriveTime', 'ActualArriveTime',
                  'ScheduledLeaveTime', 'ActualLeaveTime', 'Origin', 'Dew',
                  'Destination', 'Start.Stop',
                  'BusBunchingFlag', 'TripPatternCompleteness', 'TripLegInt',
                  ], axis=1)
    df[['Temp', 'Visibility', 'OriginLat', 'OriginLong', 'GPSLat', 'GPSLong']] \
        = df[['Temp', 'Visibility', 'OriginLat', 'OriginLong', 'GPSLat', 'GPSLong']
    ].astype('float64', copy=False)
    # Change dtype from object to interger and float. I found out that there are 'NULL'
    # values in the ScheduledHeadway,ActualHeadway,HeadwayOffset columns.
    # So using the following method with errors set to coerce option which
    # sets the value to NaN if it is NULL. So that NaNs can
    # be removed later.
    featnum = ['VehicleNo', 'BlockNo', 'ArriveLoadCompensated', 'OnsLoadCompensated',
               'OffsLoadCompensated', 'LeaveLoadCompensated', 'OnsAndOffsCompensated',
               'DwellTime', 'TravelTime', 'Humidity', 'Wind.Speed',
               'TripLeg', 'ScheduledHeadway', 'ActualHeadway', 'HeadwayOffset']
    for i in featnum:
        df[i] = pd.to_numeric(df[i], errors='coerce')
    # Replace the null values with median.
    for col in ['ScheduledHeadway', 'ActualHeadway', 'HeadwayOffset']:
        df[col] = df[col].fillna(df[col].median())
    # convert all strings to categorical values
    train_cats(df)
    # Target encoded the StopName
    from category_encoders.target_encoder import TargetEncoder

    df = df.reset_index()  # not sure why TargetEncoder needs this but it does
    targetfeatures = ['StopName']
    encoder = TargetEncoder(cols=targetfeatures)
    encoder.fit(df, df['NextLegBunchingFlag'])
    df_encoded = encoder.transform(df, df['NextLegBunchingFlag'])

    # One hot encode the DayType column
    temp = pd.get_dummies(df_encoded['DayType'])
    frames = [df_encoded, temp]
    df_encoded = pd.concat(frames, axis=1, join='outer', ignore_index=False)
    df_encoded = df_encoded.drop('DayType', axis=1)
    # Save the cleaned data in feather file
    df_encoded.to_feather(CLEANED_DATA)
