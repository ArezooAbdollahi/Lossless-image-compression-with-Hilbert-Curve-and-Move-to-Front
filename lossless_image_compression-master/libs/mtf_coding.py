import numpy as np
import random


def MTF_Encode(arr,symble_table):
    #import ipdb; ipdb.set_trace()
    st=list(symble_table)  
    n=len(arr)
    cw=np.zeros(n,dtype=int)
    for i in range(n):
        item=arr[i]
        index=st.index(item)
        cw[i]=index
        st.pop(index)
        st=[item]+st
    return list(cw)


def MTF_Decode(arr,symbol_table):
    st=list(symbol_table)  
    n=len(arr)
    data=np.zeros(n,dtype=int)
    #import ipdb; ipdb.set_trace()
    for i in range(n):
        code=arr[i]
        symbol=st[code]
        data[i]=symbol
        st.pop(code)
        st=[symbol]+ st
    return list(data)


#--------------------------test------------------------------------------------------#
def __generate_random_data(data_length,num_symbols):
    data=[]
    for i in range(data_length):
        data=data+[random.randint(0,num_symbols-1)]
    return data

if __name__=="__main__":
    print('testing mtf')
    num_of_test=5
    data_length=30
    num_symbols=10
    for i in range(num_of_test):
         print('---test no: '+str(i)+' ---')
         st=range(0,num_symbols)
         print('symble table: '+str(st))
         data=__generate_random_data(data_length,num_symbols)
         #data=[3, 4, 4, 0, 2] 
         print('plaintext   : '+str(data))
         encoded_data=MTF_Encode(data,st)
         print('encoded_data: '+str(encoded_data))
         decoded_data=MTF_Decode(encoded_data,st)
         print('decoded_data: '+str(decoded_data))
         print('------------------------------------------------------')


    

