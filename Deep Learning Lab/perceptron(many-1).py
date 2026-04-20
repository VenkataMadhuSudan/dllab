weights=[0.1,0.2,0]
input=[8.5,0.65,1.2]

output=0
assert(len(input)==len(weights))
for i in range(len(input)):
    output+=input[i]*weights[i]
print(f"Prediction is:{output:.2f}")