n=int(input("Enter the range(no. of terms):"))
if n<=0:
    print("Please enter a positive number")
else:
    a,b=0,1
    print("Fibonacci Sequence:")
    print(a,b,end=" ")
    for i in range(2,n):
        next=a+b 
        print(next,end=" ")
        a,b=b,next
    print()
    