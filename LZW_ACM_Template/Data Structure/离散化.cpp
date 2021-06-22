
//离散化模板
vector<int> v;
int scatter(vector<int> &v)
{
    sort(v.begin(), v.end());
    v.erase(unique(v.begin(), v.end()), v.end());
    return v.size();
}
inline int getid(int x)
{
    return lower_bound(v.begin(), v.end(), x) - v.begin() + 1;
}




//离散化板子  **未验证
void scatter(int a[],int n)
{
    for(int i=0;i<n;i++)
    {
        b[i]=a[i];
    }
    sort(b,b+n);
    int sz=unique(b,b+n)-b;
    for(int i=0;i<n;i++)
    {
        c[i]=lower_bound(b,b+sz,a[i])-b;
    }
}