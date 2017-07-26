# Regular Expression

标签（空格分隔）： programming_language

---
## basic token
- match any single character
`.`
- match a former character
 - `{n}` exactly n times
  `{n,}` at least n times
`{n,m}` at least n times, at most m times
 - `?`  0 or once  = {0,1}
`*`  any times  = {0,}
`+`  once or more = {1,}
- use `\` to convert
For instance, `\*` matches a `*`.
`(`, `-` need to be converted to match.
- use `?` after `+` or `*` to prevent greedness
`.*` matches as much as possible
`.*?` prevents this


- use square bracket `[]` to sepcify some characters
Eg. `[Aa]pple` matches both "Apple" and "apple".
- use `-` to show a range of alphabets or numbers
`[0-9]` specifies any single number.
`[a-zA-Z]` specifies any alphabet.

- use `^` before a group of characters to match any character except them
Eg. `appl[^e]` matches "applw" and "appl" except "apple".


- Another use of `\`: 
Transform a single character into a group
 - `\s` matches any space, including SPACE, TAB, NEXTLINE.
 - `\t` matches any non-space.
 - `\b` matches the boundary of a word.
 Eg. `er\b` matches "number" not "verb". 
 - `\B` doees **not** match the boundary of a word
 Exactly the reverse of the example above. 
 - `\d` matches a digit = `[0-9]`
 - `\D` matches a non-digit = `[^0-9]`

- Grouping
To make a sub-pattern of a pattern, grouping begins with `?:`.
Eg.  `(?:ab){1,2}` matches both "abab" and "ab". 

- tube character
 - `|` matches left **OR** right
 Eg. `a(?:b|c)d` matches both "abd" and "acd".
 
- Capturing group
extract what you group using `()`
Eg. match "aab123dd" with `ab(\d{3})d` to extract "123". 

- The beginning of a line `^`.
  The ending of a line `$`.

- Change some default settings
 - Add `g`  to do a global match `/regular expression/g`.
 - ignore case `/regular expression/i`

