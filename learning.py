import numpy
i=5#全局变量可以在函数内外使用
#全局变量在函数内不能修改
def f():
    j=5
    print(j+5)
    j=19
    '''#函数内的变量定义不影响外边的变量值'''
f()
#在函数内修改全局变量，要用关键字global
def f1():
    global i
    i=0
    print(i)
f1()
print(f.__name__)
def f2(name,age=0):
    print(f"name={name},age={age}")
f2("xiaomi")
#参数默认值要从后往前设置
#关键词参数可以打乱写f2(age=19,name="xiaowang")
#形参 parameter 实参 argument
#实参给形参赋值
def sum(*args):
    print(args)
    sum=0
    for i in args:
        sum+=i
    return sum
s=sum(1,2,3,3,2)
print(s)
#可变参数不支持关键词赋值
#可变关键词参数**kwargs,之后不能再有什么参数了
def f4(a,b,c,*args,**kwargs):
    print(a,b,c,args,kwargs)
    for k,v in kwargs.items():
        print(f'{k}:{v}')
f4(1,2,4,3,5,x='hi',y='hello')
