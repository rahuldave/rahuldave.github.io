<!-- cell:1 type:code -->
```python
#| include: false

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pandas",
#   "lxml",
# ]
# ///

```

<!-- cell:2 type:markdown -->
![PredictWise congressional voting visualization](assets/sep7.png)

<!-- cell:3 type:code -->
```python
import pandas as pd
```

<!-- cell:4 type:code -->
```python
tbl = pd.read_html("https://www.presidency.ucsb.edu/statistics/data/seats-congress-gainedlost-the-presidents-party-mid-term-elections")
```

<!-- cell:5 type:code -->
```python
df = tbl[0]
df.columns = df.columns.to_flat_index()
df
```
Output:
```
    (Unnamed: 0_level_0, Year) (Unnamed: 1_level_0, Lame Duck?)  \
0                         1934                              NaN   
1                         1938                              NaN   
2                         1942                              NaN   
3                         1946                              NaN   
4                         1950                              LD*   
5                         1954                              NaN   
6                         1958                               LD   
7                         1962                              NaN   
8                         1966                                †   
9                         1970                              NaN   
10                        1974                                ±   
11                        1978                              NaN   
12                        1982                              NaN   
13                        1986                               LD   
14                        1990                              NaN   
15                        1994                              NaN   
16                        1998                               LD   
17                        2002                              NaN   
18                        2006                               LD   
19                        2010                              NaN   
20                        2014                               LD   
21                        2018                              NaN   
22                        2022                              NaN   

   (Unnamed: 2_level_0, President) (Unnamed: 3_level_0, President's Party)  \
0            Franklin D. Roosevelt                                       D   
1            Franklin D. Roosevelt                                       D   
2            Franklin D. Roosevelt                                       D   
3                  Harry S. Truman                                       D   
4                  Harry S. Truman                                       D   
5             Dwight D. Eisenhower                                       R   
6             Dwight D. Eisenhower                                       R   
7                  John F. Kennedy                                       D   
8                Lyndon B. Johnson                                       D   
9                    Richard Nixon                                       R   
10          Gerald R. Ford (Nixon)                                       R   
11                    Jimmy Carter                                       D   
12                   Ronald Reagan                                       R   
13                   Ronald Reagan                                       R   
14                     George Bush                                       R   
15              William J. Clinton                                       D   
16              William J. Clinton                                       D   
17                  George W. Bush                                       R   
18                  George W. Bush                                       R   
19                    Barack Obama                                       D   
20                    Barack Obama                                       D   
21                 Donald J. Trump                                       R   
22            Joseph R. Biden, Jr.                                       D   

   (President's Job Approval Percentage (Gallup) As of:, Early Aug)  \
0                                                  --                 
1                                                  --                 
2                                                  74                 
3                                                  --                 
4                                                  nd                 
5                                                  67                 
6                                                  58                 
7                                                  --                 
8                                                  51                 
9                                                  55                 
10                                                 71                 
11                                                 43                 
12                                                 41                 
13                                                 --                 
14                                                 75                 
15                                                 43                 
16                                                 65                 
17                                                 --                 
18                                                 37                 
19                                                 44                 
20                                                 42                 
21                                                 41                 
22                                                 38                 

   (President's Job Approval Percentage (Gallup) As of:, Late Aug)  \
0                                                  --                
1                                                  --                
2                                                  --                
3                                                  --                
4                                                  43                
5                                                  62                
6                                                  56                
7                                                  67                
8                                                  47                
9                                                  55                
10                                                 --                
11                                                 43                
12                                                 42                
13                                                 64                
14                                                 73                
15                                                 40                
16                                                 62                
17                                                 66                
18                                                 42                
19                                                 44                
20                                                 42                
21                                                 41                
22                                                 44                

   (President's Job Approval Percentage (Gallup) As of:, Early Sep)  \
0                                                  --                 
1                                                  --                 
2                                                  74                 
3                                                  33                 
4                                                  35                 
5                                                  --                 
6                                                  56                 
7                                                  --                 
8                                                  --                 
9                                                  57                 
10                                                 66                 
11                                                 48                 
12                                                 --                 
13                                                 --                 
14                                                 54                 
15                                                 40                 
16                                                 63                 
17                                                 66                 
18                                                 39                 
19                                                 45                 
20                                                 41                 
21                                                 39                 
22                                                 44                 

   (President's Job Approval Percentage (Gallup) As of:, Late Sep)  \
0                                                  --                
1                                                  --                
2                                                  --                
3                                                  --                
4                                                  35                
5                                                  66                
6                                                  54                
7                                                  63                
8                                                  --                
9                                                  51                
10                                                 50                
11                                                 --                
12                                                 42                
13                                                 63                
14                                                 --                
15                                                 44                
16                                                 66                
17                                                 66                
18                                                 44                
19                                                 45                
20                                                 43                
21                                                 41                
22                                                 42                

   (President's Job Approval Percentage (Gallup) As of:, Early Oct)  \
0                                                  --                 
1                                                  --                 
2                                                  --                 
3                                                  --                 
4                                                  43                 
5                                                  62                 
6                                                  57                 
7                                                  --                 
8                                                  44                 
9                                                  58                 
10                                                 53                 
11                                                 49                 
12                                                 --                 
13                                                 64                 
14                                                 --                 
15                                                 43                 
16                                                 65                 
17                                                 68                 
18                                                 37                 
19                                                 45                 
20                                                 42                 
21                                                 44                 
22                                                 42                 

   (President's Job Approval Percentage (Gallup) As of:, Late Oct)  \
0                                                  --                
1                                                  60                
2                                                  --                
3                                                  27                
4                                                  41                
5                                                  --                
6                                                  --                
7                                                  61                
8                                                  44                
9                                                  --                
10                                                 --                
11                                                 45                
12                                                 42                
13                                                 --                
14                                                 57                
15                                                 48                
16                                                 65                
17                                                 67                
18                                                 37                
19                                                 45                
20                                                 41                
21                                                 44                
22                                                 40                

    (President's Party, House Seats to Defend)  \
0                                          313   
1                                          334   
2                                          267   
3                                          244   
4                                          263   
5                                          221   
6                                          203   
7                                          264   
8                                          295   
9                                          192   
10                                         192   
11                                         292   
12                                         192   
13                                         181   
14                                         175   
15                                         258   
16                                         207   
17                                         220   
18                                         233   
19                                         257   
20                                         201   
21                                         241   
22                                         222   

    (President's Party, Senate Seats to Defend)  \
0                                            14   
1                                            27   
2                                            25   
3                                            21   
4                                            21   
5                                            11   
6                                            20   
7                                            18   
8                                            21   
9                                             7   
10                                           15   
11                                           14   
12                                           12   
13                                           22   
14                                           17   
15                                           17   
16                                           18   
17                                           20   
18                                           15   
19                                           15   
20                                           20   
21                                            9   
22                                           14   

    (Seat Change, President's Party, House Seats)  \
0                                               9   
1                                             -81   
2                                             -46   
3                                             -45   
4                                             -29   
5                                             -18   
6                                             -48   
7                                              -4   
8                                             -47   
9                                             -12   
10                                            -48   
11                                            -15   
12                                            -26   
13                                             -5   
14                                             -8   
15                                            -52   
16                                              5   
17                                              8   
18                                            -30   
19                                            -63   
20                                            -13   
21                                            -40   
22                                             -9   

    (Seat Change, President's Party, Senate Seats)  
0                                                9  
1                                               -7  
2                                               -9  
3                                              -12  
4                                               -6  
5                                               -1  
6                                              -13  
7                                                3  
8                                               -4  
9                                                2  
10                                              -5  
11                                              -3  
12                                               1  
13                                              -8  
14                                              -1  
15                                              -8  
16                                               0  
17                                               2  
18                                              -6  
19                                              -6  
20                                              -9  
21                                               2  
22                                               1
```

<!-- cell:6 type:code -->
```python

```
