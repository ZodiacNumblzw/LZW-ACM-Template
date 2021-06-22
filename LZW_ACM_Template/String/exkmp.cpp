//exkmp 扩展kmp算法
const int maxn = 100010; //字符串长度最大值
int next[maxn], ex[maxn]; //ex数组即为extend数组
//预处理计算next数组
void GETNEXT(char *str)
{
    int i = 0, j, po, len = strlen(str);
    next[0] = len; //初始化next[0]，因为从0开头就可以匹配整个T，故为len   /*与EXKMP代码不同处：无这句话*/
    while(str[i] == str[i + 1] && i + 1 < len) //计算next[1]  /*与EXKMP代码不同处：比较s1和s2*/
        i++;
    next[1] = i;/*与EXKMP代码不同处：ex[0]=i*/
    po = 1; //初始化po的位置  /*与EXKMP代码不同处：po = 0*/
    for(i = 2; i < len; i++)/*与EXKMP代码不同处：i从1开始*/
    {
        if(next[i - po] + i < next[po] + po) //第一种情况，可以直接得到next[i]的值  /*与EXKMP代码不同处：next[i-po]+i<ex[po]+po*/
            next[i] = next[i - po];/*与EXKMP代码不同处：ex[i]=next[i-po];*/
        else//第二种情况，要继续匹配才能得到next[i]的值
        {
            j = next[po] + po - i;/*与EXKMP代码不同处：j=ex[po]+po-i;*/
            if(j < 0)j = 0; //如果i>po+next[po],则要从头开始匹配
            while(i + j < len && str[j] == str[j + i]) //计算next[i]  /*与EXKMP代码不同处：i+j<len&&j<l2&&s1[j+i]==s2[j]*/
                j++;
            next[i] = j;/*与EXKMP代码不同处：ex[i]=j;*/
            po = i; //更新po的位置
        }
    }
}
//计算extend数组
void EXKMP(char *s1, char *s2)
{
    int i = 0, j, po, len = strlen(s1), l2 = strlen(s2);
    GETNEXT(s2);//计算子串的next数组
    while(s1[i] == s2[i] && i < l2 && i < len) //计算ex[0]
        i++;
    ex[0] = i;
    po = 0; //初始化po的位置
    for(i = 1; i < len; i++)
    {
        if(next[i - po] + i < ex[po] + po) //第一种情况，直接可以得到ex[i]的值
            ex[i] = next[i - po];
        else//第二种情况，要继续匹配才能得到ex[i]的值
        {
            j = ex[po] + po - i;
            if(j < 0)j = 0; //如果i>ex[po]+po则要从头开始匹配
            while(i + j < len && j < l2 && s1[j + i] == s2[j]) //计算ex[i]
                j++;
            ex[i] = j;
            po = i; //更新po的位置
        }
    }
}
