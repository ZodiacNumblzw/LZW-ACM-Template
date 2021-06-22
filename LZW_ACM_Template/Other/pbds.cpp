//pbds库（平板电视）
//声明:
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>//用tree
#include<ext/pb_ds/hash_policy.hpp>//用hash
#include<ext/pb_ds/trie_policy.hpp>//用trie
#include<ext/pb_ds/priority_queue.hpp>//用priority_queue
using namespace __gnu_pbds;

//或者:
#include<bits/extc++.h>
using namespace __gnu_pbds;
//bits/extc++.h与bits/stdc++.h类似，bits/extc++.h是所有拓展库，bits/stdc++.h是所有标准库

//1.Hash
cc_hash_table<int,bool> h;  //拉链法
gp_hash_table<int,bool> h;  //探测法
//探测法快一点，和unordered_map差不多

//2.Tree
#define pii pair<int,int>
#define mp(x,y) make_pair(x,y) 
tree<pii,null_type,less<pii>,rb_tree_tag,tree_order_statistics_node_update> tr;
pii //存储的类型
null_type //无映射(低版本g++为null_mapped_type)
less<pii> //从小到大排序
rb_tree_tag //红黑树
tree_order_statistics_node_update //更新方式 
tr.insert(mp(x,y)); //插入;
tr.erase(mp(x,y)); //删除;
tr.order_of_key(pii(x,y)); //求排名 
tr.find_by_order(x); //找k小值，返回迭代器 
tr.join(b); //将b并入tr，前提是两棵树类型一样且没有重复元素 
tr.split(v,b); //分裂，key小于等于v的元素属于tr，其余的属于b
tr.lower_bound(x); //返回第一个大于等于x的元素的迭代器
tr.upper_bound(x); //返回第一个大于x的元素的迭代器
//以上所有操作的时间复杂度均为O(logn) 


//pbds还可以自己定义update结构体
//如下cf 459D
#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
template<class Node_CItr,class Node_Itr,class Cmp_Fn,class _Alloc>
struct my_node_update
{
    typedef int metadata_type;
    int order_of_key(pair<int,int> x)
    {
        int ans=0;
        Node_CItr it=node_begin();
        while(it!=node_end())
        {
            Node_CItr l=it.get_l_child();
            Node_CItr r=it.get_r_child();
            if(Cmp_Fn()(x,**it))
                it=l;
            else
            {
                ans++;
                if(l!=node_end()) ans+=l.get_metadata();
                it=r;
            }
        }
        return ans;
    }
    void operator()(Node_Itr it, Node_CItr end_it)
    {
        Node_Itr l=it.get_l_child();
        Node_Itr r=it.get_r_child();
        int left=0,right=0;
        if(l!=end_it) left =l.get_metadata();
        if(r!=end_it) right=r.get_metadata();
        const_cast<int&>(it.get_metadata())=left+right+1;
    }
    virtual Node_CItr node_begin() const = 0;
    virtual Node_CItr node_end() const = 0;
};
tree<pair<int,int>,null_type,less<pair<int,int> >,rb_tree_tag,my_node_update> me;
int main()
{
    map<int,int> cnt[2];
    int n;
    cin>>n;
    vector<int> a(n);
    for(int i=0;i<n;i++)
        cin>>a[i];
    vector<int> pre(n),suf(n);
    for(int i=0;i<n;i++)
    {
        pre[i]=cnt[0][a[i]]++;
        suf[n-i-1]=cnt[1][a[n-i-1]]++;
    }
    long long ans=0;
    for(int i=1;i<n;i++)
    {
        me.insert({pre[i-1],i-1});
        ans+=i-me.order_of_key({suf[i],i});
    }
    cout<<ans<<endl;
}



//Tire字典树
typedef trie<string,null_type,trie_string_access_traits<>,pat_trie_tag,trie_prefix_search_node_update> tr;
//第一个参数必须为字符串类型，tag也有别的tag，但pat最快，与tree相同，node_update支持自定义
tr.insert(s); //插入s 
tr.erase(s); //删除s 
tr.join(b); //将b并入tr 
pair//pair的使用如下：
pair<tr::iterator,tr::iterator> range=base.prefix_range(x);
for(tr::iterator it=range.first;it!=range.second;it++)
    cout<<*it<<' '<<endl;
//pair中第一个是起始迭代器，第二个是终止迭代器，遍历过去就可以找到所有字符串了。 
现在我们来看Astronomical Database：

#include<bits/stdc++.h>
#include<ext/pb_ds/assoc_container.hpp>
#include<ext/pb_ds/trie_policy.hpp>
using namespace std;
using namespace __gnu_pbds; 
typedef trie<string,null_type,trie_string_access_traits<>,pat_trie_tag,trie_prefix_search_node_update>pref_trie;
int main()
{
    pref_trie base;
    base.insert("sun");
    string x;
    while(cin>>x)
    {
        if(x[0]=='?')
        {
            cout<<x.substr(1)<<endl;
            auto range=base.prefix_range(x.substr(1));
            int t=0;
            for(auto it=range.first;t<20 && it!=range.second;it++,t++)
                cout<<"  "<<*it<<endl;
        }
        else
            base.insert(x.substr(1));
    }
}


//其余的比如优先队列，STL里面已经有了就不再赘述