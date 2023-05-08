1. file_type的枚举类型解释：
- status_error: 获取文件类型时出现错误。
- file_not_found: 文件不存在。
- regular_file: 常规文件。
- directory_file: 目录文件。
- symlink_file: 符号链接文件。
- block_file: 块设备文件。
- character_file: 字符设备文件。
- fifo_file: 命名管道（FIFO）文件。
- socket_file: 套接字文件。
- type_unknown: 文件类型未知。

2. space_info结构解释：
capacity表示存储介质的总容量，free表示存储介质中的可用空间，available表示存储介质中当前可供使用的空间。

3. perms的枚举类型解释：
- no_perms：文件或目录没有权限。
- owner_read：文件或目录的所有者具有读取权限。
- owner_write：文件或目录的所有者具有写入权限。
- owner_exe：文件或目录的所有者具有执行权限。
- owner_all：文件或目录的所有者具有所有权限（读、写和执行）。
- group_read：文件或目录的群组成员具有读取权限。
- group_write：文件或目录的群组成员具有写入权限。
- group_exe：文件或目录的群组成员具有执行权限。
- group_all：文件或目录的群组成员具有所有权限。
- others_read：其他用户（即非所有者或群组成员）具有读取权限。
- others_write：其他用户具有写入权限。
- others_exe：其他用户具有执行权限。
- others_all：其他用户具有所有权限。
- all_read：所有用户（即所有者、群组成员和其他用户）具有读取权限。
- all_write：所有用户具有写入权限。
- all_exe：所有用户具有执行权限。
- all_all：所有用户具有所有权限。
- set_uid_on_exe：如果设置了这个位，文件或目录将使用其所有者的特权来执行，而不是使用执行它的用户的特权。
- set_gid_on_exe：如果设置了这个位，文件或目录将使用其群组的特权来执行，而不是使用执行它的用户的特权。
- sticky_bit：如果设置了这个位，文件或目录只能被其所有者、父目录的所有者或超级用户删除或重命名。
- all_perms：所有权限都被设置（包括set-UID、set-GID和sticky位）。
- perms_not_known：权限未知。

