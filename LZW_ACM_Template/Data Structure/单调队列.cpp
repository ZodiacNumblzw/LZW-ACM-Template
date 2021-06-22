//单调队列    (deque慎用，不如手写)
void mono_queue(int n,int k)
{
    deque<pair<int,int> >q;
    for(int i=1; i<=n; i++)
    {
        int m=a[i];
        while(!q.empty()&&q.back().second<m)
            q.pop_back();
        q.push_back(pair<int,int>(i,m));
        if(q.front().first<=i-k)
            q.pop_front();
    }
}