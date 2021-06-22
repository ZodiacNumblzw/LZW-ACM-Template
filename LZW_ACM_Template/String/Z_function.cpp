int z[maxn];
int s[maxn];
void ZF(int n)
{
    for(int i=1,l=0,r=0; i<n; i++)
    {
        z[i] = i>r ? 0 : min(r-i+1, z[i-l]);
        while(i+z[i]<n && s[z[i]]==s[i+z[i]]) ++z[i];
        if(i + z[i] - 1 > r) l=i, r=i+z[i]-1;
    }
}

//Z函数
vector<int> z_function(string s) {
  int n = (int)s.length();
  vector<int> z(n);
  for (int i = 1, l = 0, r = 0; i < n; ++i) {
    if (i <= r) z[i] = min(r - i + 1, z[i - l]);
    while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
    if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
  }
  return z;
}


//第二种写法
int z[M]; //z函数，z[i]表示字符串s的第i个后缀和s的lcp，不包括0位置。
void exkmp(int *s)
{
    for(int i=1,l=0,r=0; s[i]; ++i)
    {
        z[i] = i>r ? 0 : min(r-i+1, z[i-l]);
        while(s[i+z[i]] && s[z[i]]==s[i+z[i]]) ++z[i];
        if(i + z[i] - 1 > r) l=i, r=i+z[i]-1;
    }
}