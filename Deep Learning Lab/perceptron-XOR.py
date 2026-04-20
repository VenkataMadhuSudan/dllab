# XOR gate using NAND gate and perceptron 

w1,w2,b=0.5,0.5,-1
learning_rate,epochs=0.1,100
inputs=[(0,0),(0,1),(1,0),(1,1)]
des_outputs=[1,1,1,0]

def activate(x):
    return 1 if x>=0 else 0

for epoch in range(epochs):
    total_error=0
    for i in range(len(inputs)):
        A,B=inputs[i]
        tar_output=des_outputs[i]
        output=activate(w1*A+w2*B+b)
        error=tar_output-output
        w1+=learning_rate*error*A
        w2+=learning_rate*error*B
        b+=learning_rate*error
        total_error+=abs(error)
    if total_error==0:
        break

print("XOR gate:")
for A,B in inputs:
    mid1=activate(w1*A+w2*B+b)
    mid2=activate(w1*A+w2*mid1+b)
    mid3=activate(w1*mid1+w2*B+b)
    output=activate(w1*mid2+w2*mid3+b)
    print(f"Inputs:({A},{B}) Output:({output})")
