weights=[ [0.1,0.1,-0.3],
          [0.1,0.2,0.0],
          [0.0,1.3,0.1]]
input=[8.5,0.65,1.2]

output=[0,0,0]
for i in range(len(weights)):
    for j in range(len(input)):
        output[i]+=input[j]*weights[i][j]
print("Prediction is:")
print(output)

    