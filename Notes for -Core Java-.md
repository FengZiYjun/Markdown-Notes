# Notes for *Core Java* 

标签（空格分隔）： programming_language

---
Author: Cay S. Horstmann
Volumn I - Fundamentals, Ninth Edition 


before the notes: 
From C++ to Java there are a lot of details that need to be paid attention to. And this book explains them quite clearly. 

## Chapter Three: Fundamental programming structures in Java
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

- When two values are combined with a binary operator, both operands need to be converted. 
 - If one of the two operands is `double` or `float` or `long`, the other one will be converted into the same type. 
 - Otherwise, both will be converted into an `int`. 

### Casts
the same as C++ 
`int x = (int)9.9;`
Boolean type can not be cast into any numeric type. 
an alternative to conversion:  `b ? 1 : 0`

### String

- definition
`String str = "hello";`
- methods
 - `String substring(int beginIndex, int endIndex)` get the range of [beginIndex, endIndex) 
`String substring(int beginIndex)`
  - Concatenation using `+`
 When a string object is concatenated with a value that is not a string, it will be converted into a string, since every Java object can be converted into a string. 
 - Testing for equality: `str.equals(another_string)` 
Even `"hello".equals(str)` is valid. 
`str.equalsIgnoreCase()`
The == operator is only used to determine whether two strings are stored in the same location. 
To test whether a string is `null`, use `if(str==null)`.
**Attention**: `null` is different from empty string "" with the length of zero. 
 - The `str.length()` method yields the number of code units(char) required to make up the string object. 
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
 - `String trim()` eliminates all leading and trailing spaces 
  
- Building Strings 
use `StringBuilder` class
```
StringBuilder builder = new StringBuilder();
builder.append("abc");
String str = builder.toString();
```

- A few things about Java String:
1. Strings are immutable
You cannot change a character in an existing string. 
An advantage of immutable strings: strings can be shared. 
2. Java String is analogous to a `char*` pointer rather than `char[]` in C programming and similar to `string` objects in C++ programming, whose memory management is performed by constructors, assignment operators and destructors. But C++ strings are mutable with every char inside modifiable. 
Java does automatic garbage collection. 


### Input & Output

- Reading input 
1. construct a `scanner` that is attached to System.in
`Scanner in = new Scanner(System.in);`
2. To read a line, use nextline() method.
`String line = in.nextline()`
To read a word, use `next()`
To read a integer, use `nextIn()`
`nextDouble()` etc.
`boolean hasNext()`
`boolean hasNextInt()` 

To read a password, there is a little different. 
```
Console cons = System.console();  // a console object for interaction
String name = cons.readline("User name: ");
char[] password = cons.readPassword("Password: ");
// for security reason, returns an array rather than a string. 
```

- Formatting output
`System.out.print()`
Java SE 5.0 brings this from the C library: 
`System.out.printf("%d, %.5f", age, salary);`
Similar use in constructing a String: 
`String str = String.format("%d, %.5f", age, salary);`

 - conversion characters
 mostly the same as C
`printf("%tc", new Date());` There are many different formats to print a date. 
 - flags
1. The flag `$` is used to specify the index(start from 1, not 0) of the argument to be formatted. 
`printf("%2$d, %1$tc, %2$f", date, salary);` 
print the salary in decimal format and then in floating-point format. print the date in time format. 
2. The flag `<` is used to indicate the same argument as the previous. 
`printf("%ty, %<tm, %<td", new Date());`

- File Input & Output
 - `Scanner in = new Scanner(Paths.get("myfile.txt"));`
Do not directly put a String as Scanner's parameter.
The file should be located in the relative directory of where JVM starts. To see it, use `System.getProperty(user.dir)`. Otherwise, use an absolute directory like `C:\\User\\lenovo`.

 - `PrintWriter out = new PrintWriter("myfile.txt");`
If the file is not found, it will be created. 


### Control Flow
Mostly they are identical to C++ or C, but there are slight differences. 
The same things: if, while, for, switch 

1. Java has no `goto`, but a labeled `break`. 
Put the label with a colon before the loop. After the break, the control flow will jump across the whole loop.
2. Redefining a variable inside a nested block is not allowed. In C++, the inner one shadows the outer one. 

### Big Numbers
in java.math package, use `BigInteger` and `BigDecimal`(floating-point). 

- Methods: 
`static BigInteger valueOf(long x)`
`BigInteger add(BigInteger other)` the same as substract, multiply, divide, mod
`int compareTo(BigInteger other)` do the substract to get the result

**Attention**: Java does not allow programmable operator overloading. So you cannot use + or - to big numbers. 


### Arrays
An array is a date structure that stores a collection of values of the same type. You can access through index like `a[1]`. 

- definition 
`int[] a = new int[100];`
which is similar to `int* a` in C++
 `int a[]` is also valid but not recommended.

- initialization
When you created an array of numbers/boolean/objects, all elements are initialized with 0/false/null.
(null means it does not hold any object.)
Different ways of initialization: 
`int[] a = {1,2,3};`
`int[] a = new int[]{1,2,3};`
An array with the length of zero is OK, but it is not null.

- for-each loop
`for(type variable: collection) statement`
sets the given variable to each element of the collection and excecute the statement. 
The collection must be an array or objects of a class that implements the `Iterable` interface. 

- String array in command-line parameters
`public static void main(String[] args)` 
receives an array of Strings specified in the command line. 
call the program with the following command: 
`java class_name -q one two`
Then `-q`, `one` and `two` are args[0], args[1] and args[2] respectively. 


- Methods
 - `toString()` returns a string like `"[1, 2, 3, 4]"`
 - `Array.copyOf(type[], int length)` for deep copy
`Array.copyOf(type[], int start, int end)`
  Simply assigning the name of the old array to the new one is the shallow copy.
 - `static void sort(type[] array)`
 using QuickSort algorithm to sort
 - `static int binarySearch(type[] a, int start, int end, type v)`
 using binary search for sorted arrays. If found, index is returned. Otherwise, a negative r is returned and -r-1 is where it should be inserted. 
 - `static void fill(type[] a, type v)`
 set all element of the array to v
 - `static boolean equals(type[] a, type[] b)` 
 
- Multidimensional Array
`double[][] a = new double[row][col];`
`double[][] b = {{1,2},{3,4}};`
access by two pairs of brackets [][]

 - using for-each: 
```
for(double[] row: a){
    for(double value: row)
        ...
}
```
 - To print as a String, use
`Array.deepToString(b)`
and outputs something like `"[[1,2],[3,4]]"`
 - the same definition in C++ would be: 
`double** a = new double*[row];`
`for(int i=0;i<row;i++)`
   ` a[i] = new double[col];`


## Chapter Four: Objects and Classes
### OOP
Niklaus Wirth, the designer of Pascal language said, 
> Algorithm + Data structure = Programs 

This is the traditional understanding of programming, algorithm first and then data structure. But in object-oriented programming, data comes first, and then look at the algorithms.

### Classes
####Concepts:

- class, object, instance, instance field, method, state
- encapsulation, inheritance

#### Objects
Three things about an object: 

- behavior: What can I do with the object? What are its methods? 
- state: How it react when its methods invoked? 
- identity: How to distinguish from others that have the same behavior and state? 

#### indentify classes 
Nouns are class. Verbs are methods. 
More commonly, experience-based. 

#### relationship between classes 

- dependency - "uses-a"
- aggregation - "has-a" 
- inheritance - "is-a" 

UML notation

#### Object Variable
An object variable does not contain an object. It only refers to an object. 

Java object variables are analogous to C++ object pointers. 
`Date birthday; // Java`
 is the same as 
`Date* birthday; // C++`

accessor: get methods
mutator: set methods

### Customer-defined Classes
One source file has only one public class and any number of nonpublic class. The name of the source file must match that of the public class. 
The compiler will create `.class` file for each class. 

Recommendation: make all instance field private, except for public final fields. 

### Constructors
A constructor 

- has the same name as the class
- can take any number of parameters
- has no return value
- is always called with `new` operator
- A class can have more than one constructors.

One of the common error for C++ programmer to code in Java is forgetting the `new` when creating an object. 

Be careful not to introduce local variables inside constructors with the same name as instance fields. 

### Implicit and explicit parameters
Explicit parameters are explicitly listed in the declearation of the method, while the implicit one is the object of type that appears before the method name. 
The key word `this`can be used to refer to the implicit parameter.

Unlike C++, all methods of Java classes are defined inside the class. JVM decides which method is inline. 

### Encapulation
To get and set the value of an instance field, we need

- a private data field
- a public field accessor method 
- a public field mutator method

If an accessor wants to return references to mutable objects, it should be cloned first by `.clone()` method. Otherwise encapsulation will be breaken. 

Access privileges are class-based, not instance-based. 
Therefore, a method can access the private data of all objects of this class. 

### Final 
Final instance fields must be initialized when the object is constructed. Having set the final field value must be guaranteed after the end of every constructor. 

If an object variable is declared as final, it does not mean that the object itself is constant but the object reference that stored in the variable does not change after construction. 

### Static 
If a field is defined `static`, then there is only one such field per class. 
Static variables are rare. Static constants are more common. 
And public constants are OK because no one can modify. 
`public static final `

Static methods ar eused in two situations: 

- no need to access object state and all parameters are explicit.
- only need to access the static field of a class

### main 
The main method does not operate on any object. Every class can have a main method. It is a handy trick for unit testing. 

### Method parameters 
Java always uses **call by value**.
Methods get a copy of all parameter values passed to it and cannot modify the contents of any parameter variables. 

A method

- cannot modify a parameter of a primitive type(numbericc or boolean).
- cannot make an object parameter refer to a new object
- can change the state of an object parameter

### Object Constructiion 
Constructors can be overloaded.
Default field initialization.
There is a free no-argument constructor only when my class has no other constructors.

- explicit field initilization 
simply assign a value in the class definition
This assignment will be carried out **before the constructor**. 

- construtor calls constructor
If the first statement of a constructor has the form of `this()`, it calls another constructor of the same class. 
It never happends in C++.

- initializatino blocks
set assingment statements of fields inside a block
denote `static` before the block if initialization is complex. 
not common


- order of initialization 
1. All data fields are initialized to default values.
2. Explicit field initializers and initialization blocks are excecuted in declaration order. 
3. A constructor is executed. 

### Destruction
Java does automatic garbage collection. 
A `finalize` method will be called before it. But do not rely on it for resource recycling. 
Instead, supply a `close` method that does the cleanup. 

### Package
A package is a collection of classes. 
The standard Java package is inside the java and javax package hierachies. 
The main reason for using package is to guarantee the uniqueness of class names. Classes in different packages can have the same name.
To guarantee the uniqueness of package name, use your domain name in reverse. 
There is no relationship between nested packages. 

### Importation 
A class can use all classes from its own package and all public classes from other package. 
To access classes from other packages, use its full name or use `import` for short. 

The package and import in Java is analogous to the namespace and using in C++, rather than #include. 

#### Static import
`import static` + package_name
To use the static methods andd fields of the class without prefix

####　Addition of a class into a package 
put the package statement at the top of the source file
`package com.mycompany.corejave;`
Otherwise, it will be in the default package. 

To complile in the command line: 
`javac com/mycompany/corejave/wedget.java`
`java com.mycompany.corejave.wedget`
The compiler looks for files, while the interpreter looks for classes, which accounts for the differ of their paths.

If the feature(class, method, variable) does not have public or private modifier, it is package-visible. Classes from the same package can access it. This will break encapsulation. 


package sealing: no further classes can be added. (Chapter 10)


### class path 
Class files can be stored in a JAR(Java Archive) file which contains multiple class files adn subdirectories in a compressed ZIP format. 


To share classes among programmers, 
1. Place class files inside a directory. 
2. Place any JAR files inside another directory. 
3. Set the *class path* - The base directory, the JAR file and the current directory (.)

The compiler always looks for files in the current directory. But JVM only looks into the current directory if the "." is on the class path.

ways for setting the class path: 
1. using command `java -classpath` + the three paths
2. set CLASSPATH environment variable in different shells
Do not set CLASSPATH permanently!

### Documentation Comments
`javadoc` is a tool in JDK that generates HTML documentation from source files. 

#### Insertion 
`javadoc` looks for 

- Packages 
- Public classes and interfaces 
- Pubic and protected fields.
- Public and protected constructors and methods 

The comment is placed above the feature it described.
Start with `/**`, end with `*/`
Tags start with `@`
You can use HTML modifiers. 


1. Class Comments
must be placed after any import statements, before the class definition. 

2. Method Comments 

- `@param` variable_name discription
- `@return` description
- `@throws` class description

3. Field Comments 
Only need to document public field - static constants. 

4. General Comments 

- `@author` name
- `@version` text 
- `@see` reference

5. Package Comments
supply an HTML file named `package.html` and all text in its body will be extracted. 

To generate documentation, run 
`javadoc -d ` docDirectory nameOfPackage1 nameOfPackage2 ...

### Class Design Hints

- Always keep data private. 
- Always initialize data. 
- Do not use too many basic types in a class. (group them into a new class)
- Not all fields need individual field accessors and mutators. 
- Break up classes that have too many responsibilities.
- Make the names of your classes and methods reflect their responsibilities. 




## Chapter 5: Inheritance 

### Superclasses and subclasses

- using the key word `extends` to inherite: 
`class SubClass extends SuperClass{ ... }`
All inhetitance in Java is public inheritance. 
Subclass has no direct access to the private field of the superclass. 

- A subclass can define method that overrides the same one in the superclass. 
To call the superclass method from the subclass, use `super.`method_name()

- To call the superclass constructor from the subclass constructor: `super(`*parameter list* `);`

In C++, we use `class derived: public base{ ... }` to inheritate, `base::method()` to call base methods, `Derived(init_list): Base(init_list){ ... }` to construct the part of the superclass.

**Polymorphism**: An object variable can refer to multiple actual types.
**Dynamic binding**: Automatically select the appropriate method at runtime. 

Dynamic binding is default in Java. You don't have to declare `virtual` as C++. 

Java does not support multiple inheritance. Use interface instead. 

Overriding can change the return type of a method to a subtype of the orinal one. 

The subclass method that overrides that of the superclass must be declared as public as well. 

- using `final` to ban inheritance
A `final` method cannot be overrided. 
A `final` class cannot be extended. All methods in the class are final, but not the fields. 

###Casting

- Casting is only available within the inheritance hierarchy. 
- `instance of` is used to check before casting from superclass to subclass. 

```
if(x instance of SubClass){ // false if x is null
    SubClass sub = (SubClass)x;
}

// in c++
SubClass* sub = dynamic_cast<SubClass*>(x);
if(sub != NULL){
```

### Abstract Classes
- A class with at least one abstract method must be defined as an abstract class itself. 
- But abstract classes can have common methods and fields as well.
- Abstract methods give place to subclasses to implement them. 
- A class can be defined as  abstract even though it has no abstract methods. (?)
- An abstract class cannot be instantiated. 

When a abstract class is extended, there are two choices: 

- leave at least one abstract method undefined, and the subclass is abstract as well. 
- implement all abstract methods in the superclass. 

In C++, a class is abstract if it has at least one pure virtual function such as  `virtual void f() = 0;`

### Protected Access
use with caution in fields
make more sense to methods

protected in Java is less safe than in C++, because 
other classes in the same package can access it. 

### Summary of accessibility 
1. `private` visible to the class only
2. `public` visible to the world
3. `protected` visible to the package and all subclasses
4. (default) visible to the package


### The Cosmic Superclass: Object
Every class in Java extends Object.
In Java, only premitive types (numbers, characters and boolean) are not objects. 

#### Equality Testing
testing equality between a subclass and a superclass
two senarios: 

- If a subclass can have its own notion of equality, the symmetry rules forced you to use `getClass` test - `x.equals(y)` returns true if and only if `y.equals(x)` returns true. 
- If the notion of equality is fixed in the superclass, use the `instanceof` test and allow objects of different subclasses to be equal to one another. 

The recipe for `equals` method: 

1. the explicit parameter for the method is `Object otherObject`.
And this will override the one of the Object class.
2. Test whether `this` happens to be identical to otherObject.
3. Test whether otherObject is null.
4. Compare the class of `this` and otherObject.
 If the sematics of `equals` can change in subclasses, use the `getClass` test.
   If the same sematics holds for all subclasses, use the `instanceof` test. 
5. Cast otherObject to a variable of your class type.
6. Compare the fields. Use == for primitive types and `Object.equals` for objects. 
7. If you redefine `equals` in a subclass, include a call to `super.equals`

```
public boolean equals(Object otherObject){
    // 2. 
    if(this == otherObject) return true;
    // 3. 
    if(otherObject == null) return false;
    // 4. 
    if(getClass() != otherObject.getClass()) return false;
    // or
    if(!(otherObject instanceof ClassName)) return false;
    // 5.
    ClassName other = (ClassName)otherObject;
    // 6. 
    return field1 == other.field1
        && Object.equals(field2, other.field2)
        && ...;
}
```

#### The *hashcode* method
The *hashcode* method is defined in the Object class. 

- For String class, the hash code is derived from their contents. Therefore two string variables with the same contents will have the same hash code. 
- For other classes, the hash code is derived from the memory address. 

If *equals* method is redefined, *hashcode* method needs to be redefined as well. And they must be compatible. 
If two variables are equals under the *equals* method, they must have the same value of hash code. 

Some relative methods: 
`int Object.hashcode()` returns the hash code for this object (returns 0 for null). 
`int Object.hash(Object...Object)` returns a combined hash code.
`itn Array.hashcode(type[] a)` computes the hash code of array a with different components.

#### The *toString* method
A common format for *toString* method: 
`getClass().getName()` followed by field values in square brackets. For derived classes, add square brackets after. 

The *toString* method is invoked automatically when contatenating a string with an object by "+" operator ,or printing an object by `System.out.println()`.

Specifically, to print an array, use `Array.toString(a)` instead. 
For a multimensional array, use `Array.deepToString(a)`

Adding a *toString* to a user-defined class is strongly recommended. 

### Generic Array List 
ArrayList is a **generic** class with a type parameter in angle brackets. 
ArrayList is similar to the C++ vector template. 

- construction
`ArrayList<Type> al = new ArrayList<Type>();`
In Java 7, "diamond syntax" can omit the type parameter on the right.
`ArrayList<Type> al = new ArrayList<>();`
Or with initialized capacity:
`ArrayList<Type> al = new ArrayList<>(100);`

- operations
`boolean add(Type obj)` append, always return true.
`int size()` 
`void ensureCapacity(int capacity);` allocate an array with desired storage capacity

- accessing elements
No [] syntax can be used. 
`Type get(int index)`
`void set(int index, Type obj)`
`Type remove(int index)` shifts down all element above and returns the removed element

- warning 
ArrayList has a "raw" version which takes no type paramters. 
The "raw" one cannot be assigned or cast into a typed one.
The compliler would not check if you pass a typed ArrayList into a "raw" ArrayList and this is dangerous. 

### Object Wrappers and Autoboxing 
All primitive types have class counterparts. 
They are called *wrappers*：
Integer, Long, Float, Double, Short, Byte, Character, Void, Boolean
*Wrapper* classes are **immutable** and **final**. 
A typical usage is to construct an ArrayList of integer. 
Angle brackets do not recieve primitive types. 
use `ArrayList<Interger> list = new ArrayList<>()` instead. 

Autoboxing
`list.add(3)` does autoboxing to be `list.add(Integer.valueOf(3))`
In most cases, the primitive types and their wrappers are likely to be the same except for their identity. 
Wrappers are a convenient place to put some basic methods. 
The word "boxing" is taken from C#. 

`int Integer.intValue()` return as an int
`static String Integer.toString()`
`static int Integer.parseInt(String s, int radix)` returns the integer contained in a string. The integer should be in the given base or default 10 base. 
`static Integer Integer.valueOf(String s, int radix)` similar

### Varargs Methods: methods with a variable number of parameters
`public void f(int... list){ }`
The ellipsis "..." is part of the code, which denotes the method receives an arbitary number of objects, and is exactly the same as `Object[]`. So `for(int x: list)` can be used for iteration. 

To call the method, use `f(1,2,3);`.

### Enumeration classes
Enumeration is actually a class with a fixed number of instances. 
`public enum MyEnum{SMALL, MEDIAN, LARGE };`

### Reflection 
The Reflection Library provides tools for manipulating Java codes dynamically. A program that can analyze the capabilities of classes is called *reflective*. 
It is a powerful and complex mechanism. 
However, it is useful in system programming or toolkit design, not in applications.

#### The *class* class 
The Java runtime system maintains *runtime type identification* on all objects, which keeps track on the class which each object belongs to.

- Get the class infomation from an instance 
`class cl = x.getClass();`
- `getName()` method is commonly used, but it works strange for array types (historical reason).
- To obtain a class object by its name 
`Class cl = Class.forName("java.util.Date");`
Or simply
`Class cl = Date.class;`

- You can use "==" to compare class objects. 
- To create a new instance of the class, use `x.getClass().newInstance();`, which non-parameter constructor will be invoked. (cannot pass any parameters)

#### A primer on Catching Exceptions 

- Checked Exception: The compiler checks whether you provide a handler 
- Unchecked Exception: Programmers should avoid these mistakes rather than coding handlers.

Using "try-catch" statement to catch exceptions.
The compiler will tell you which method need to supply a handler. 

#### Using refection 
The three classes `Field`, `Method`, `Constructor` in Java.lang.reflect package describe three aspects of the class respectively. 
`getName` returns a name
`getModifier` returns an integer
`Modifier.isPublic/isPrivate/...` analyze the integer, return boolean
`Field.getType`
`getFields`, `getMethods`, `getConstructors` return arrays of the *public* fields, methods and constructors. 
`getDeclaredFields`, `getDeclaredMethods`, `getDeclaredConstructors` return all, not just public ones.


#### Using Reflecton for analyzing objects at runtime
access field information by the `get` method of the `Field` class

```
MyClass mc = new MyClass();
Class cl = new mc.getClass();
Field f = cl.getDeclaredField("field_name");
Object obj = f.get(mc);
// returns an object whose value is the current value of the field of mc
```
This method can only be used to access accessible fields. If the field is private, you can change its accessibility by `setAccessible` method on a `Field`, `Method`, `Constructor` object. 
`f.setAccessible(true);`

As for number objects, `get()` does autoboxing. 

Also, we can set value by `set` method 
`void Field.set(Object obj, Object newValue);`

> customize your own generic toString method used for all objects

#### Using reflectio to write Generic Array Code
Consider we are coding the implementation of the `copyOf` method of a geneic array that holds elements of any type.
Any object can be converted into `Object` class, but it would not help because an array of elements of `Object` class would not be cast back into the same type of array after the memory reallocation. 
Therefore we would not cast an instance of `MyClass[]` into `Object[]` class. Rather, we treat the instance of `MyClass[]` as an Object and create a new array using `reflect.Array.newInstance()` based on this Object.
```
public static Object copyOf(Object obj, int newLength){
    Class cl = obj.getClass();
    if(!cl.isArray()) return null;
    Class componentType = cl.getComponentType();
    int length = Array.getLength(cl);
    Object newArray = Array.newInstance(componentType, newLength);
    System.arraycopy(obj, 0, newArray, 0, Math.min(length, newLength));
    return newArray;
}

int[] a = {1,2,3,4};
a = (int[])copyOf(a, 10);
```

#### Invoking arbitrary methods
Java has no method pointer because the designer thought it is dangerous and Java interface is an alternative. 
Howeverm the reflection mechanism allows you to call arbitrary methods. Yet In a mush slower way. 

- obtain a `Method` object
`Method getMethod(String name, Class... parameterTypes)`
`Method m = x.class.getMethod("method_name",int.class);`
- the method class has an `invoke` method
`Object invoke(Object obj, Object... args)`
The first parameter is the implicit parameter, and the rest are explicit.
For static methods, the first parameter "obj" should be null. 
 
Suggestion: use method class only when absolutely necessary. 

### Design Hints for Inheritance 
1. place common operations and fields in the superclass.
2. Do not use protected fields. 
3. Use inheritance to model the "is-a" relationship.
4. Do not use inheritance unless all inherited methods make sense. 
5. Do not change the expected behavior when overriding a method. 
6. Use polymorphism, not type information. 
7. Do not overuse reflection. 


## Chapter Six: Interfacces and Inner Classes 
Two advanced techniques: 

- *interface*  describe what classes should do, regardless of how they should do it.
- *inner classes* help design collections of cooperating classes. 
