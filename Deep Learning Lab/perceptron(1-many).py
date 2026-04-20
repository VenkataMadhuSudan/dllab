weights=[0.3,0.2,0.9]
input=0.65

output=[0,0,0]
assert(len(output)==len(weights))
for i in range(len(weights)):
    output[i]=input*weights[i]
print("Prediction is:")
print(output)



     