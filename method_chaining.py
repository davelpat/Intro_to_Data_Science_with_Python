"""
Suppose we are working on a DataFrame that holds information on our equipment for an upcoming backpacking trip.
Can you use method chaining to modify the DataFrame df in one statement to 
drop any entries where 'Quantity' is 0 and rename the column 'Weight' to 'Weight (oz.)'?
"""

# reproduce the dataframe to play with
d = ({'Item': ['Pack', 'Tent', 'Sleeping Pad', 'Sleeping Bag', 'Toothbrush/Toothpaste', 'Sunscreen', 'Medical Kit', 'Spoon', 'Stove', 'Water Filter', 'Water Bottles', 'Pack Liner', 'Stuff Sack', 'Trekking Poles', 'Rain Poncho', 'Shoes', 'Hat'],
      'Category': ['Pack', 'Shelter', 'Sleep', 'Sleep', 'Health', 'Health', 'Health', 'Kitchen', 'Kitchen', 'Kitchen', 'Kitchen', 'Utility', 'Utility', 'Utility', 'Clothing', 'Clothing', 'Clothing'],
      'Quantity': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1], 
      'Weight (oz.)': [33.0, 80.0, 27.0, 20.0, 2.0, 5.0, 3.7, 0.7, 20.0, 1.8, 35.0, 1.0, 1.0, 16.0, 6.0, 12.0, 2.5]})
qdf_base = pd.DataFrame(data=d).set_index('Item')
qdf_base

                       Category  Quantity  Weight (oz.)
Item                                                   
Pack                       Pack         1          33.0
Tent                    Shelter         1          80.0
Sleeping Pad              Sleep         1          27.0
Sleeping Bag              Sleep         1          20.0
Toothbrush/Toothpaste    Health         1           2.0
Sunscreen                Health         1           5.0
Medical Kit              Health         1           3.7
Spoon                   Kitchen         1           0.7
Stove                   Kitchen         1          20.0
Water Filter            Kitchen         1           1.8
Water Bottles           Kitchen         2          35.0
Pack Liner              Utility         1           1.0
Stuff Sack              Utility         1           1.0
Trekking Poles          Utility         1          16.0
Rain Poncho            Clothing         1           6.0
Shoes                  Clothing         1          12.0
Hat                    Clothing         1           2.5

#==== Method chaining
# for the method chaining exercise, 'Sleeping Pad' qty was set to 0
qdf = qdf_base.copy()
answer = qdf.where(qdf["Quantity"] > 0).dropna().rename(columns={'Weight': 'Weight (oz.)'})
answer

# Course solution -- only the head of the dataframe is defined here using a different method
qdf = pd.DataFrame([{"Item" : "Pack", "Category" : "Pack", "Quantity" : 1, "Weight" : 33.0}, {"Item" : "Tent", "Category" : "Shelter", "Quantity" : 1, "Weight" : 80.0}, {"Item" : "Sleeping Pad", "Category" : "Sleep", "Quantity" : 0, "Weight" : 27.0}, {"Item" : "Sleeping Bag", "Category" : "Sleep", "Quantity" : 1, "Weight" : 20.0}, {"Item" : "Toothbrush/Toothpaste", "Category" : "Health", "Quantity" : 1, "Weight" : 2.0}])
qdf.set_index('Item', inplace=True)
print(qdf.drop(qdf[qdf['Quantity'] == 0].index).rename(columns={'Weight': 'Weight (oz.)'}))


#==== Groupby
"""
Looking at our backpacking equipment DataFrame, suppose we are interested in finding our total weight for each category. Use groupby to group the dataframe, and apply a function to calculate the total weight (Weight x Quantity) by category.
"""

qdf = qdf_base.copy()
qdf['Agg Weight'] = qdf['Weight (oz.)'] * qdf['Quantity']
qdf.groupby('Category').agg({'Agg Weight': np.sum})

# course solution
qdf = qdf_base.copy()
qdf.groupby('Category').apply(lambda qdf,a,b: sum(df[a] * qdf[b]), 'Weight (oz.)', 'Quantity')


# Or alternatively without using a lambda:
# def totalweight(df, w, q):
#        return sum(df[w] * df[q])
#        
# print(df.groupby('Category').apply(totalweight, 'Weight (oz.)', 'Quantity'))   


#==== Pivot tables
"""
Suppose we have a DataFrame with price and ratings for different bikes, broken down by manufacturer and type of bicycle.

Create a pivot table that shows the mean price and mean rating for every 'Manufacturer' / 'Bike Type' combination.
"""

b = ({'Bike Type': ['Mountain', 'Mountain', 'Road', 'Road', 'Mountain', 'Mountain', 'Road', 'Road', 'Mountain', 'Mountain', 'Road', 'Road'],      
      'Manufacturer': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
      'Price': [400, 600, 400, 450, 300, 250, 400, 500, 400, 500, 800, 950], 
      'Rating': [8, 9, 4, 4, 6, 5, 4, 6, 5, 6, 9, 10]})
bdf = pd.DataFrame(data=b)
bdf

   Bike Type Manufacturer  Price  Rating
0   Mountain            A    400       8
1   Mountain            A    600       9
2       Road            A    400       4
3       Road            A    450       4
4   Mountain            B    300       6
5   Mountain            B    250       5
6       Road            B    400       4
7       Road            B    500       6
8   Mountain            C    400       5
9   Mountain            C    500       6
10      Road            C    800       9
11      Road            C    950      10

bdf.pivot_table(values=['Price', 'Rating'], index='Manufacturer', columns='Bike Type', aggfunc=np.mean)

# Course solution
Bikes = bdf
pd.pivot_table(Bikes, index=['Manufacturer','Bike Type'])
