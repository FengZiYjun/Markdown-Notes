# Vim notes
### 1. changes between three modes
i or a : ordinary mode --> insertion mode
':' ordinary mode --> command line mode
Esc: other modes --> ordinary mode
### 2. exit vim
in command mode 
`:wq` or `:x` save and quit
`:q` quit
`:q!` forced quit
`:wq!` forced save and quit

in ordinary mode 
`exit` or shift+zz

### 3. enter vim
in command line `$ vim <filename>`
or in vim command line mode `:e <filepath>`

### 4. basic operation in ordinary mode 
h j k l ---> left down up right
w/b ---> next word/previous word

### 5. different insertion
i --> edit at the current cursor
I/A --> insert at the front/back of the line
O/o --> insert a new line before/after the current line

### 6. save the file
in command line mode
`w` save 
`:w <filename>` or `:saveas <filepath>` save as 
`wq` save and quit

### 7. delete
in ordinary mode
- char
`x`
- word
`dw`
- lines
`dd` delete a line
`3dd` delete 3 lines
`dG` delete to end
`d1G` delete to top

### 8. repeat
`.` repeat the previous action
`number<command>` repeat the command for 'number' times

### 9. cursor shift
- among lines
`nG` to the nth line
`gg` to the first line
`G` to the last line
- to the previous position
ctrl + o
- inside the line
`0`/`^` to the line head
`$` to the line tail
`w`/`e` to the beginning/end of the next word
`b`/`ge` to the beginning/end of the previous word
`f<letter>` search backward and shift to the first matching position

### 10. copy & paste & cut
in ordinary mode
- copy `y`
`yy` copy a line  `3yy` 3 lines
`yw` copy a word  `y3w` 3 words
`y0` copy to the line head
`y$` copy to the line tail
`yG` copy to the buttom
`y1G` copy to the top
- paste `p`
- cut `dd` the same as delete
- exchange adjacent lines `ddp`

### 11. replace & undo
`r<lettter>`  replace a letter at the cursor
`cc` delete a line and enter insertion mode
`cw` delete a word and enter insertion mode
`~` capital to non-capital
`u` undo  `u3` undo 3 times
ctrl+r redo

###　12. indented
in ordinary mode
`>>` right indented
`<<` back to left
in command line mode
`:set shiftwidth=3` change setting
`:ce` make the line in the center of file
`le`/`:ri` at the left/right of the file

### 13. search
in ordinary mode
`/<string><Enter>`
`/` search downward  `?` search upward
`n` continue search 
`N` reverse search

## advanced
>命令行模式下输入:set autoindent(ai) 设置自动缩进
命令行模式下输入:set autowrite(aw) 设置自动存档，默认未打开
命令行模式下输入:set background=dark或light，设置背景风格
命令行模式下输入:set backup(bk) 设置自动备份，默认未打开
命令行模式下输入: set cindent(cin) 设置C语言风格缩进
>
 

