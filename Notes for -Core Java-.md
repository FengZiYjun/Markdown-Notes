# Notes for *Core Java* 

标签（空格分隔）： programming_language

---
Volumn I - Fundamentals, Ninth Edition 

before the notes: 
From C++ to Java there are a lot of details that need to be paid attention to. And this book explains them quite clearly. 

## Chapter Three: Fundamental programming structure in Java
### Data type

- Integer
**byte** 1 byte
**short** 2 bytes
**int** 4 bytes
**long** 8 bytes
Java has no "unsigned".
- Floating-Point
**float** 4 bytes
**double** 8 bytes
To check whether x is "Not a number": 
`Double.isNaN(x)`
- Char
do not to use`char` unless there is actual need to manipulate UTF-16 code units. 
- Boolean
You cannot convert between integer and boolean. 
### Variables
Variable names are case-sensitive. 
The length of variable names are unlimited. 
In Java, no declaration is seperated from definition. 
e.g. such things like `extern int i` will not happen in Java. 
### Constants
use the key word `final` to denote a constant. 
use `static final` inside a class to denote a **class constant**.
### Operators
+, -, *, /, +=
++, --
==, !=
&&, ||
? : 
bitwise operators: & | ^ ~
math functions and constants: 
`Math.sqrt(x)`
`Math.PI`
### Conversions between numeric types
- without loss precision: 
byte---short---int---long/double
char---int

- with loss precision: 
int---float
long---double 
long---float

- When two values are conbined with a binary operator, both operands need to be converted. 
 - if one of the two operands is `double` or `float` or `long`, the other one will be converted into the same type. 
 - Otherwise, both will be converted into an `int` 

### Casts
same as C++ 
`int x = (int)9.9;`
boolean type can not be cast into any numeric type. 
an alternative to conversion:  `b ? 1 : 0`

### String

- definition
`String str = "hello";`
- methods
 - `String substring(int beginIndex, int endIndex)` get the range of [beginIndex, endIndex) 
`String substring(int beginIndex)`
  - concatenation using `+`
 When a string object is concatenated with a value that is not a string, it will be converted into a string, since every Java object can be converted into a string. 
 - testing for equality: `str.equals(another_string)` 
Even `"hello".equals(str)` is valid. 
`str.equalsIgnoreCase()`
The == operator is only used to determine whether two strings are stored in the same locaion. 
To test whether a string is `null`, use `if(str==null)`.
Attention: `null` is different from empty string "" with the length of zero. 
 - the `str.length()` method yields the number of code units(char) required to make up the string object. 
To get the true length - the number of code pointers, use `str.codePointCount(0,str.lenth())`
 - `int codePointAt(int index)` return the code point that starts or ends at certian location. 
`str.charAt(n)` return the code unit at position n which begin at 0, but it is too low-level. Do not use it. 
 - `int offsetByCodePoints(int startIndex, int cpCount)`
returns the index of the code point away from the code point at startIndex.
 - `int str.compareTo(String other)`
returns a nagetive if `str` comes before `other` in dictionary order and a positive if `str` comes after `other`, or 0 when the two are equal.
 - `boolean endsWith(String suffix)`
`boolean startWith(String prefix)`
  - `int indexOf(String str)`
`int indexOf(String str, int fromIndex)`
`int inedxOf(int cp)`
`int inedxOf(int cp, int fromIndex)`
returns the start of the last substring equal to the string str or the code point cp, starting at the end of the string or at fromIndex. 
 - `Strin replace(String oldString, String newString)`
returns a new String by replacing all substring `oldString` with `newString`. 
 - `String toLowerCase()`
`String toUpperCase()`
 - `String trim()` eliminate all leading and trailing spaces 
  
      
- A few things about Java String:
1. Strings are immutable
You cannot change a character in an existing string. 
An advantage of immutable strings: strings can be shared. 
2. Java String is analogous to a `char*` pointer rather than `char[]` in C programming and similar to `string` objects in C++ programming, whose memory management is performed by constructors, assignment operators and destructors. But C++ strings are mutable with every char inside modifiable. 
Java does automatic garbage collection. 

 

