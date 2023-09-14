**GIT** 

(software)**:** DISTRIBUTED VERSION CONTROL SYSTEM.

Everyone has one copy of repo,can be used for any type of files,save the changes,stores history of changes and multiple people can work on different features of same project/application/software.

Using command line gives control

After installing git,first config name and email for others to know who is committing.

![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.001.png)

epo cloning: gets all code from server to pc

Repo initialize: start from start

Initialization:

Creates a git repository:

*myfir\_000@Home MINGW64 ~/Desktop/git and github*

*$ **git init***

*Initialized empty Git repository in C:/Users/myfir\_000/Desktop/git and github/.git/*

Checks files: -a shows hidden files too  **.git**/ stores all data and files, no need to open:

*myfir\_000@Home MINGW64 ~/Desktop/git and github (master)*

*$ **ls -lart***

*total 4*

*drwxr-xr-x 1 myfir\_000 197121 0 Aug 21 23:04 ../*

*drwxr-xr-x 1 myfir\_000 197121 0 Aug 21 23:15 ./*

*drwxr-xr-x 1 myfir\_000 197121 0 Aug 21 23:15 .git/*

**git status** shows status about various files

**touch**  command  creates a file with no content

![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.002.png)

**git add**  command adds given file into staged and keeps track of its commits.

**git commit -m [message explaining the commit]**

This command commits the changes and -m is used to write a message for the commit so people can understand why this commit was done.

If only command **git commit**  is written it directs to a vym editor, type I to write the message for commit and then press esc and tye **:wq** to exit.

**git add -A** add all files to staging area

**git log** shows commit history



**FILES:**

**UNTRACKED:** git doesn't keep track of those files

**STAGED:** after adding files,git keeps record of their commits/changes

**COMMIT:** used to record changes and keep track

Committed files go to unmodified and they can also be modified

TO START A PROJECT FIRST COMMIT IS NECESSARY.

IN TERMINAL UP DOWN ARROW CAN BE USED TO VIEW COMMAND HISTORY


![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.003.png)


AFTER MAKING ANY CHANGES,ADDING FILE TO STAGING AREA IS NECESSARY AND THEN COMMIT

To zoom in and out the terminal **ctrl + /ctrl -**



![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.004.png)

**git checkout [filename] :** matches the file to the last commit of working directory

Used to recover the changes

**git checkout -f** matches all files to previous commit

**git log -p -n** will display history and changes of last n commits

**git diff** compares working directory with staging area,shows difference(changes) with the two

` `If we make changes and add the file to staging area,it wont show anything as now working directling is same as staging area

**git diff –staged** (2 hypens) compares staging area with last commit

Diff tells the changes which were made after last add/commit

**git commit -a -m [message]** commits changing directly,no need to add to staging area

**git rm [filename]** deletes the file and removes from git staging area

**git rm –cached [filename]** (2 hyphen) : remove file from git staging area but doesnt delete from hard disk(file goes to untracked)

**git status -s (**short status) gives overview in short

![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.005.png)

Here 2 boxes before filename shows staging area(green) and working tree(red) 

Here 3 files were modified so showed red M(modified)

Then contact was added so showed green M as added to staging area

Then again contact was modified so it showed green M,modified in staging area wrt last commit and a red M as modified in working tree wrt staging area

**.gitignore** cant be created in windows directly,only possible with help of terminal

1\.Contain name of files which are to be ignored by git for add/push/commit/pull

2\. In .gitignore file if we mention **\*.log** all files with extension .log will be ignored

3\. If file mentioned as **/filename** then file with that will be ignored only for that working tree containing .ignore file, other files with that name wont be ignored

4\. .gitignore can be mentioned in itself to be ignored

5\. To ignore a whole directory mention **foldername/** in .gitignore


**BRANCHES:** creates copy,doesnt affect other branches,master/main is the head/root branch of all

**git branch [branch name] :** creates a branch inside current branch

**git branch:** displays name of all branches and shows \* on current branch name with green colour

**git checkout [branch name]**: switches to the specified branch

All laters changes/commits will be on specified branch

![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.006.png)

Here changes are made in sub branch feature1 but when **git checkout master**  is runned,files as per master branch are restored and changes are erased

**git checkout feature1:** will bring back to sub branch 

And its changes will be restored.

When we run **git merge [sub branch]** in the master branch,its changes will be applied in master branch

If we don't merge and run git log commits of sub won't be  shown in master but will be shown after merging

**git checkout -b [sub branch name]** will create branch in current branch and switch to newly created branch for further execution.

**GITHUB:**

Hosting service for git repositories

Local repos are hosted on remote repos

**git remote add origin [remote url] :** connects local repo to remote

origin is the name[reference] for the url

**git remote**: shows all remote repos

**git remote** -v: shows remote urls for fetching and pushing

**git remote set-url origin [url link]** sets a new ur for origin

**git push -u [destination repo(origin)] [source branch(master)]** pushes the commits and files of mentioned branch to github origin repo

**git push** pushes to the previous source destination mentioned after -u

**PULL** request to the repo owner to change the code

![](Aspose.Words.a4e1a24f-e7f6-433d-aa1e-32defc426645.007.png)

**SSH KEYS:**GIVE ACCESS OF GITHUB ACCOUNT TO OUR COMPUTER(for private repo on github)

**FOR CLONING ANY PUBLIC REPO:** copy its url from github and in pc folder run the command

**git clone [url] [foldername]** url repo will be cloned to foldername

If foldername is not mentioned it will create a folder with same name as on github


