w1,b=0.1,-1
learning_rate,epochs=0.1,100
inputs=[0,1]
des_outputs=[1,0]
def activate(x):
    return 1 if x>=0 else 0

for epoch in range(epochs):
    total_error=0
    for i in range(len(inputs)):
        A=inputs[i]
        tar_output=des_outputs[i]
        output=activate(w1*A+b)
        error=tar_output-output
        w1+=learning_rate*error*A
        b+=learning_rate*error
        total_error+=abs(error)
    if total_error==0:
        break

print("NOT gate:")
for A in inputs:
    output=activate(w1*A+b)
    print(f"Input:({A}) Output:({output})")