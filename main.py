from ast import Index
from cmath import nan
from decimal import Decimal
from numbers import Number
from tokenize import Double
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from matplotlib import pyplot as plt
from keras import backend as K
#from tensorflow.keras import layers

#from matplotlib import pyplot as plt

# The following lines adjust the granularity of reporting.
#@title Define the plotting function

def plot_the_loss_curve(epochs, mse):
  """Plot a curve of loss vs. epoch."""

  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, mse, label="Loss")
  plt.legend()
  plt.ylim([mse.min().values[0]*0.95,mse.max().values[0]*1.03])
  plt.show()  

print("Defined the plot_the_loss_curve function.")

#@title Define the functions that build and train a model
def build_model(my_learning_rate,feature_layer):
  """Create and compile a simple linear regression model."""
  # Most simple tf.keras models are sequential.
  model = tf.keras.models.Sequential()

  model.add(feature_layer)

  model.add(tf.keras.layers.Dense(units=20, activation='relu',name='Hidden1'))

  model.add(tf.keras.layers.Dense(units=12, activation='relu',name='Hidden2'))
  
  model.add(tf.keras.layers.Dense(units=1,name='Output'))

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])
                
  return model               


def train_model(model, dataset,epochs,label_name,batch_size=None):
    features = {name:np.asarray(value).astype(np.float32) for name,value in dataset.items()}
    label = np.asarray(features.pop(label_name)).astype(np.float32)
    history = model.fit(x=features,y=label,batch_size=batch_size,epochs=epochs,shuffle=True)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist[["root_mean_squared_error"]]
    return epochs,mse

def SlopData(df):
    shape = df[["開市"]].shape
    for index in range(shape[0]):
        value1 = 0
        if isinstance(df.loc[index,"收市"],str):
            value1 = float(df.loc[index,"收市"].replace(',',''))
        else:
            value1 = float(df.loc[index,"收市"])

        value2 = 0

        if isinstance(df.loc[index,"開市"],str):
            value2 = float(df.loc[index,"開市"].replace(',',''))
        else:
            value2 = float(df.loc[index,"開市"])  
            
        newValue = value1 - value2
        df.loc[index,"收市"] = newValue
        index += 1
    return df

def MidPriceData(df):
    shape = df[["開市"]].shape
    for index in range(shape[0]):

        value1 = 0

        if isinstance(df.loc[index,"收市"],str):
            value1 = float(df.loc[index,"收市"].replace(',',''))
        else:
            value1 = float(df.loc[index,"收市"])

        value2 = 0

        if isinstance(df.loc[index,"開市"],str):
            value2 = float(df.loc[index,"開市"].replace(',',''))
        else:
            value2 = float(df.loc[index,"開市"])  

        newValue = (value2 + value1) / 2
        df.loc[index,"收市"] = newValue
        index += 1
    return df

def MidPriceDataMuti(Days,df):
    index = 0
    total = 0
    shape = df[["收市"]].shape
    for dfIndex in range(shape[0]):
        index += 1
        if index > Days :
            index -= 1
            total += df.loc[dfIndex,"收市"]
            total -= df.loc[dfIndex - Days,"收市"]
            df.loc[dfIndex,"收市"] = total / Days
        else:
            total += df.loc[dfIndex,"收市"]
                
        df.loc[dfIndex,"收市"] = total / index
    return df

    
def MidPriceDataAdd(Days,df):
    index = 0
    total = 0
    shape = df[["收市"]].shape
    for dfIndex in range(shape[0]):
        index += 1
        if index > Days :
            index -= 1
            total += df.loc[dfIndex,"收市"]
            total -= df.loc[dfIndex - Days,"收市"]
            df.loc[dfIndex,"收市"] = total / Days
        else:
            total += df.loc[dfIndex,"收市"]
    return df

def CheckNanByType(data,data_isnull,type):
    
    shape = data[type].shape
    for index in range(shape[0]):
        if bool(data_isnull.loc[index,type]):
            whileIndex = index
            while bool(data_isnull.loc[whileIndex,type]):
                whileIndex -= 1
                if whileIndex >= 0:
                    data.loc[index,type] = data.loc[whileIndex,type]
                else:
                    data.loc[index,type] = 0
                    break
        index += 1
    return data

def CheckNan(data):
    
    data_isnull = data.isnull()
    CheckNanByType(data,data_isnull,"收市")
    CheckNanByType(data,data_isnull,"開市")
    CheckNanByType(data,data_isnull,"高")
    CheckNanByType(data,data_isnull,"低")
    CheckNanByType(data,data_isnull,"成交量")
    return data

def CheckMissingData(data,sample_data):
    shape = sample_data["日期"].shape
    for index in range(shape[0]):
        fitData = data.loc[data["日期"] == sample_data.loc[index,"日期"]] 
        if fitData["日期"].size == 0:
            newRow : dict[str:any]
            if index > 0 and index >= data["日期"].shape[0]:
                newRow = {
                    '日期':[sample_data.loc[index,"日期"]],
                    '收市':[data.loc[data["收市"].size - 1,"收市"]],
                    '開市':[data.loc[data["開市"].size - 1,"開市"]], 
                    '高':[data.loc[data["高"].size - 1,"高"]],
                    '低':[data.loc[data["低"].size - 1,"低"]],
                    '成交量':[data.loc[data["成交量"].size - 1,"成交量"]]
                }
            elif index > 0:
                newRow = {
                    '日期':[sample_data.loc[index,"日期"]],
                    '收市':[data.loc[index - 1,"收市"]],
                    '開市':[data.loc[index - 1,"開市"]], 
                    '高':[data.loc[index - 1,"高"]],
                    '低':[data.loc[index - 1,"低"]],
                    '成交量':[data.loc[index - 1,"成交量"]]
                }
            else:
                newRow = {
                    '日期':[sample_data.loc[index,"日期"]],
                    '收市':[data.loc[index + 1,"收市"]],
                    '開市':[data.loc[index + 1,"開市"]], 
                    '高':[data.loc[index + 1,"高"]],
                    '低':[data.loc[index + 1,"低"]],
                    '成交量':[data.loc[index + 1,"成交量"]]
                }
            newDataFrame = pd.DataFrame(newRow,index=[data['日期'].shape[0]])
            data = pd.concat([data,newDataFrame],axis=0)
            index += 1
    return data
def InsertData(data,type,newData,newType):
    shape = data["日期"].shape
    for index in range(shape[0]):
        data.loc[index,type] = newData.loc[index,newType]
        index += 1
    return data
def MoveNextData(data,type):
    shape = data["日期"].shape
    for index in range(shape[0] - 1):
        data.loc[index,type] = data.loc[index + 1,type]
        index += 1
    return data
def CheckStringK(data):
    shape = data["日期"].shape
    for index in range(shape[0]):
        if isinstance(data.loc[index,'成交量'],str):
            if data.loc[index,'成交量'].find('K') >= 0:
                data.loc[index,'成交量'] = float(data.loc[index,'成交量'].replace('K',''))*1000
        index += 1
    return data
def CrossFeatures(train_df_norm,types):
    resolution_in_Zs = 0.3
    typeList = []

    for type in types:
        train_df_norm_Column_Type1 = tf.feature_column.numeric_column(type)
        Type1_Boundaries = list(np.arange(int(min(train_df_norm[type])),int(max(train_df_norm[type])),resolution_in_Zs))
        Type1 = tf.feature_column.bucketized_column(train_df_norm_Column_Type1,Type1_Boundaries)
        typeList.append(Type1)

    Type1_x_Type2 = tf.feature_column.crossed_column(typeList,hash_bucket_size=100)
    cross_feature = tf.feature_column.indicator_column(Type1_x_Type2)
    return cross_feature
def ValueSet(gold_train_df,USD_train_df,SP_train_df,WTI_train_df):
    CheckNan(gold_train_df)
    CheckNan(USD_train_df)
    CheckNan(SP_train_df)
    CheckNan(WTI_train_df)

    CheckStringK(gold_train_df)
    CheckStringK(USD_train_df)
    CheckStringK(SP_train_df)
    CheckStringK(WTI_train_df)

    gold_train_df.drop(labels=['升跌（%）'],axis=1,inplace=True)
    USD_train_df.drop(labels=['升跌（%）'],axis=1,inplace=True)
    SP_train_df.drop(labels=['升跌（%）'],axis=1,inplace=True)
    WTI_train_df.drop(labels=['升跌（%）'],axis=1,inplace=True)

    gold_train_df = CheckMissingData(gold_train_df,USD_train_df)
    gold_train_df = CheckMissingData(gold_train_df,SP_train_df)
    gold_train_df = CheckMissingData(gold_train_df,WTI_train_df)
    USD_train_df = CheckMissingData(USD_train_df,gold_train_df)
    SP_train_df = CheckMissingData(SP_train_df,gold_train_df)
    WTI_train_df = CheckMissingData(WTI_train_df,gold_train_df)

    gold_train_df.sort_values(by=['日期'],inplace=True)
    USD_train_df.sort_values(by=['日期'],inplace=True)
    SP_train_df.sort_values(by=['日期'],inplace=True)
    WTI_train_df.sort_values(by=['日期'],inplace=True)
    SP_slop = SlopData(SP_train_df)[['收市']]
    SP_Mid_Price = MidPriceData(SP_train_df)[['收市']]
    SP_Mid_Price_30 = MidPriceDataMuti(30,SP_Mid_Price)[['收市']]
    SP_Mid_Price_7 = MidPriceDataMuti(7,SP_Mid_Price)[['收市']]
    SP_Mid_Price_100 = MidPriceDataMuti(100,SP_Mid_Price)[['收市']]
    SP_slop_7 = MidPriceDataAdd(7,SP_slop)[['收市']]
    SP_slop_30 = MidPriceDataAdd(30,SP_slop)[['收市']]
    SP_slop_100 = MidPriceDataAdd(100,SP_slop)[['收市']]

    WTI_slop = SlopData(WTI_train_df)
    WTI_Mid_Price = MidPriceData(WTI_train_df)
    WTI_Mid_Price_30 = MidPriceDataMuti(30,WTI_Mid_Price)
    WTI_Mid_Price_7 = MidPriceDataMuti(7,WTI_Mid_Price)
    WTI_Mid_Price_100 = MidPriceDataMuti(100,WTI_Mid_Price)
    WTI_slop_7 = MidPriceDataAdd(7,WTI_slop)
    WTI_slop_30 = MidPriceDataAdd(30,WTI_slop)
    WTI_slop_100 = MidPriceDataAdd(100,WTI_slop)

    gold_slop = SlopData(gold_train_df)
    USD_slop = SlopData(USD_train_df)
    gold_Mid_Price = MidPriceData(gold_train_df)
    USD_Mid_Price = MidPriceData(USD_train_df)
        
    gold_Mid_Price_100 = MidPriceDataMuti(100,gold_Mid_Price)
    USD_Mid_Price_100 = MidPriceDataMuti(100,USD_Mid_Price)

    gold_Mid_Price_30 = MidPriceDataMuti(30,gold_Mid_Price)
    USD_Mid_Price_30 = MidPriceDataMuti(30,USD_Mid_Price)
    gold_Mid_Price_7 = MidPriceDataMuti(7,gold_Mid_Price)
    USD_Mid_Price_7 = MidPriceDataMuti(7,USD_Mid_Price)

    gold_slop_100 = MidPriceDataAdd(100,gold_slop)
    USD_slop_100 = MidPriceDataAdd(100,USD_slop)

    gold_slop_30 = MidPriceDataAdd(30,gold_slop)
    USD_slop_30 = MidPriceDataAdd(30,USD_slop)
    gold_slop_7 = MidPriceDataAdd(7,gold_slop)
    USD_slop_7 = MidPriceDataAdd(7,USD_slop)

    #gold_train_df = gold_train_df[["高","低"]]
    gold_train_df = InsertData(gold_train_df,"SPSLOP100",SP_slop_100,"收市")
    gold_train_df = InsertData(gold_train_df,"SPSLOP30",SP_slop_30,"收市")
    gold_train_df = InsertData(gold_train_df,"SPSLOP7",SP_slop_7,"收市")
    gold_train_df = InsertData(gold_train_df,"SP100",SP_Mid_Price_100,"收市")
    gold_train_df = InsertData(gold_train_df,"SP30",SP_Mid_Price_30,"收市")
    gold_train_df = InsertData(gold_train_df,"SP7",SP_Mid_Price_7,"收市")
    gold_train_df = InsertData(gold_train_df,"SPSLOP",SP_slop,"收市")
    gold_train_df = InsertData(gold_train_df,"SP",SP_Mid_Price,"收市")


    gold_train_df = InsertData(gold_train_df,"GOLDSLOP100",gold_slop_100,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLDSLOP30",gold_slop_30,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLDSLOP7",gold_slop_7,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLD",gold_Mid_Price,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLDSLOP",gold_slop,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLD100",gold_Mid_Price_100,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLD30",gold_Mid_Price_30,"收市")
    gold_train_df = InsertData(gold_train_df,"GOLD7",gold_Mid_Price_7,"收市")

    gold_train_df = InsertData(gold_train_df,"USD100",USD_Mid_Price_100,"收市")
    gold_train_df = InsertData(gold_train_df,"USD30",USD_Mid_Price_30,"收市")
    gold_train_df = InsertData(gold_train_df,"USD7",USD_Mid_Price_7,"收市")
    gold_train_df = InsertData(gold_train_df,"USDSLOP100",USD_slop_100,"收市")
    gold_train_df = InsertData(gold_train_df,"USDSLOP30",USD_slop_30,"收市")
    gold_train_df = InsertData(gold_train_df,"USDSLOP7",USD_slop_7,"收市")
    gold_train_df = InsertData(gold_train_df,"USDSLOP",USD_slop,"收市")
    gold_train_df = InsertData(gold_train_df,"USD",USD_Mid_Price,"收市")

    gold_train_df = InsertData(gold_train_df,"WTI100",WTI_Mid_Price_100,"收市")
    gold_train_df = InsertData(gold_train_df,"WTI30",WTI_Mid_Price_30,"收市")
    gold_train_df = InsertData(gold_train_df,"WTI7",WTI_Mid_Price_7,"收市")
    gold_train_df = InsertData(gold_train_df,"WTISLOP100",WTI_slop_100,"收市")
    gold_train_df = InsertData(gold_train_df,"WTISLOP30",WTI_slop_30,"收市")
    gold_train_df = InsertData(gold_train_df,"WTISLOP7",WTI_slop_7,"收市")
    gold_train_df = InsertData(gold_train_df,"WTISLOP",WTI_slop,"收市")
    gold_train_df = InsertData(gold_train_df,"WTI",WTI_Mid_Price,"收市")

    MoveNextData(gold_train_df,"GOLDSLOP")
    
    gold_train_df.fillna(0)
    gold_train_df.drop(labels=['日期'],axis=1,inplace=True)
    return gold_train_df

pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
pd.set_option('display.max_columns', None)

tf.keras.backend.set_floatx('float32')

print(tf.version.VERSION)
print("Imported the modules.")

gold_train_df = pd.read_csv("Gold_History.csv")
USD_train_df = pd.read_csv("USD_History.csv")
SP_train_df = pd.read_csv("SP_History.csv")
WTI_train_df = pd.read_csv("WTI_History.csv")

gold_test_df = pd.read_csv("Gold_History_Test.csv")
USD_test_df = pd.read_csv("USD_History_Test.csv")
SP_test_df = pd.read_csv("SP_History_Test.csv")
WTI_test_df = pd.read_csv("WTI_History_Test.csv")

train_data = ValueSet(gold_train_df,USD_train_df,SP_train_df,WTI_train_df)
shuffled_train_df = train_data.reindex(np.random.permutation(train_data.index))
shuffled_train_df_mean = shuffled_train_df.mean()
shuffled_train_df_std = shuffled_train_df.std()
shuffled_train_df_norm = (shuffled_train_df - shuffled_train_df_mean)/shuffled_train_df_std

test_data = ValueSet(gold_test_df,USD_test_df,SP_test_df,WTI_test_df)
test_df = test_data.reindex(np.random.permutation(test_data.index))
test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean)/test_df_std

my_feature = ["SPSLOP100","SPSLOP30","SPSLOP7","SP100","SP30","SP7","SPSLOP","SP",
"GOLDSLOP100","GOLDSLOP30","GOLDSLOP7","GOLD","GOLD100","GOLD30","GOLD7",
"USD100","USD30","USD7","USDSLOP100","USDSLOP30","USDSLOP7","USDSLOP","USD",
"WTI100","WTI30","WTI7","WTISLOP100","WTISLOP30","WTISLOP7","WTISLOP","WTI"]

my_label = "GOLDSLOP" # the median house value on a specific city block.

feature_columns = []

for i in range(0,len(my_feature)):
    feature_columns.append(tf.feature_column.numeric_column(my_feature[i]))
for i in range(0,len(my_feature) - 1):
    for s in range(i + 1,len(my_feature)):
        feature_columns.append(CrossFeatures(shuffled_train_df_norm,[my_feature[i],my_feature[s]]))
myfeature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 20
batch_size = 1000

# That is, you're going to create a model that predicts house value based 
# solely on the neighborhood's median income.  

# Invoke the functions to build and train the model.
my_model = build_model(learning_rate,myfeature_layer)
epochs, mse = train_model(my_model, shuffled_train_df_norm, epochs,my_label, batch_size)
plot_the_loss_curve(epochs, mse)

learning_rate = 0.01
epochs = 1
batch_size = 1

test_features = {name:np.asarray(value).astype(np.float32) for name,value in test_df_norm.items()}
test_label = np.array(test_features.pop(my_label))
my_model.evaluate(x=test_features,y=test_label,batch_size=batch_size)

tf.saved_model.save(my_model, "saved/model")
