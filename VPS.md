#  VPS

标签（空格分隔）： 未分类

---
VPS(Virtual Private Server) is generally a remote server which allows users to do a lot of things.

1. Buy a server. 
Several companies provide VPS at reasonable prices. I bought one from *Bandwagon*(Official website: https://bandwagonhost.com/), which costs me nearly $3 per month. 
After paying for the VPS, customers should recieve an **IP address**  which specifies the server they allocate for you, and an **SSH port** which is used to access the server. 
2. Obtain control.
*Bandwagon* provides a control panel for the server, called KiwiVM. Login your *Bandwagon* account to enter it. 
In KiwiVM, you can obtain a **root password** of the VPS.
KiwiVM has the full control of your server so it is important to set a KiwiVM password to protect it.
(Some companies tell you the root password together with the IP address in an email or something else. But Bandwagon does not do so for security reasons.)
3. Initialization.
In KiwiVM, you can do a lot of initialization for your server. For example, start or shutdown the server, see disk or memory usage, install a new operating system, etc.
- I installed Ubuntu 16 on my server. 
- The name of the server is the **host name**, which can be any name you like. 

4. SSH connection. 
Secure Shell (SSH) is a cryptographic network protocol for operating network services securely over an unsecured network.
In this case, the **host (machine)** is the computer I am using. The operating system on the host is Windows.
*PuTTy* is one of the softwares that helps you establish SSH connection. Download the compatible Windows version and install.
Open *PuTTy*. Fill in the IP address and Port. Choose SSH as the connection type. Then a black panel appears in which you enter the word "host" and the root password to login. After that you are successfully connected to the server.

5. Basic Security. 
Immediately, you should change your root password the first time you access the server. Using the command
`passwd`
to change a new password that no one knows but you.
Generally it is insequre to work with the root account which has the full priledge of the system. 
To create a new user of this server, type 
`adduser`
to specify the name of the user and its password, both of which are required to login.
To create a new user with `sudo` previledge, type
`visudo`
and add the user name in the proper position of the shown file.

6. Build Shadowsocks.
Access the root account. Type
`apt install shadowsocks`
to install.
Enter this directory.
`cd /etc/shadowsocks`
Edit the configuration file.
`vim config.json`
Enter the IP address of the server and create a new password for shadowsocks.
To start the shadowsocks, type
`nohup ssserver -c /etc/shadowsocks/config.json`
Then in client end (Windows) you can establish the connection with shadowsocks of the Windows version by filling in the IP address and the purticular password.
Therefore the client can access blocked websites.
7. Transfer Files.
- From Windows client to Ubuntu server
Download `pscp.exe` from *Putty* offical website to your Windows. It is a tool for file transfer.
cd to where the `pscp.exe` is. 
`pscp file_path_on_client user_name@ip_address:path_of_server`
This is the basic command to transfer files from local Windows to the remote server. 
However, since we do not use the usual port for SSH, the server port needs to be specified by `-P 29818`.
So the command looks like this:
`pscp -P 29818 file_path_on_client user_name@ip_address:path_of_server`
Then the user password is required.

- From Ubuntu server to Windows client
Similarily, 
`pscp -P 29818 user_name@ip_address:path_of_server file_path_on_client`
The user password is required.

These operations are done in Windows command line. If you want to transfer files to Windows in Ubuntu command, a SSH server is needed for Windows.
