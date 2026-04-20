w1,w2,b = 0.5,0.5,-1
def activate(x): 
    return 1 if x >= 0 else 0
inputs = [(0,0),(0,1),(1,0),(1,1)]
des_outputs = [0,0,0,1]
lr, epochs = 0.1, 100
for _ in range(epochs):
    error_sum = 0
    for (A,B), t_out in zip(inputs, des_outputs):
        out = activate(w1*A + w2*B + b)
        e = t_out - out
        w1 += lr * e * A
        w2 += lr * e * B
        b  += lr * e
        error_sum += abs(e)
    if error_sum == 0:
        break
print("AND Gate:")
for A,B in inputs:   
    output=activate(w1*A+w2*B+b)
    print(f"Input:({A},{B}) Output:{output}")
