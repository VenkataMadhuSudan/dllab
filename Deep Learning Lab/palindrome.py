s=input("Enter a string:").lower()

is_palindrome=True
for i in range(len(s)//2):
    if s[i]!=s[len(s)-i-1]:
        is_palindrome=False
        break
if is_palindrome:
    print("It is a palindrome")
else:
    print("It is not a palindrome")

    