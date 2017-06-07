# Git Basic
### 1. initial configuration
`git config -global user.name "?"`
`git config -global user.email "?"``
to check
`cat ~/.gitconfig`

### 2. get a git reposary
#### 2.1 clone
`git clone http://git.?.com/?/project_name`
go into the reposary
`cd ".../project_name/"`
(in window shell file path must be quoted in ")

#### 2.2 build
chose a place
`cd "/my dictionary/"`
check the files in this dictionary
`ls`
make a folder
`madir project_name`
`cd "project"`
build a git 
`git init`
A empty git reposary called .git will be created 

### 3. work flow
To creat new files
`touch file1 file2 file3`

To edit the file with vim or just 
`echo "Hello" >> file1`

To show the content of the file
`cat file1`

To check the status
`git status`

To check the modification in current path
`git diff`
To check the difference between local buffer and the last commit 
`git diff --cache` 

To add a new file to local Index(buffer)
`git add <filename>`
`git add *` add all new and modified files

To add the file to local git reposary
`git commit -m "commit_massage"`

To synchronize local reposary to remote git reposary
`git push origin master`
`git push <url>`

### 4. branch and merge
To build a branch (b1 is a branch name)
`git branch b1`

To check all branches
`git branch`
The asterisk* indicates the branch I am woking in.

To switch between branches
`git checkout b1`
`git checkout master`

To see the difference of branches
`git diff master b1`
`git diff master` in b1 branch

To merge a branch into master
`git checkout master`
`git merge -m "merge b1 to master" b1`
The merge will fail if a file was modified differently in two branches

To delete a branch
`git branch -d b1`

To undo a merge
`git reset --hard HEAD^`

### 5. Git Log
To check all the commit
`git log`
A file will be open. Display the file by pressing "Enter" and quit by pressing "q"

more display of Log
`git log --graph`
`git log --stat`
`git log --pretty=short` medium full fuller email raw

### 6. Distributive Working Flow
To clone a local reposary
`git clone /home/.../project`
And you will have two reposaries with absolutely the same content. Simply call them
pj and pj_clone.
while you have done some edition in pj_clone, how to renew pj?
`cd /.../pj ` in pj's dictionary
`git pull /home/.../pj_clone`
pull the modification of another reposary into the current reposary.

Knowing that pj_clone is cloned from pj, so pj_clone will renew modification of pj after this command in pj_clone master
`git pull`

