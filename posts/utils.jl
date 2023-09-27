using RData,DataFrames, CodecBzip2,Pipe,GLM,GLMakie,PrettyTables
using Combinatorics,ColorSchemes,RCall

"""
    load_rda(str::AbstractString)
    加载 Stat2 rda  dataset
"""
function load_rda(str::AbstractString)
 df=load("../../Stat2Data/$str.rda")
 return df["$str"]
end

"构建 Stat2  Struct"
Base.@kwdef struct  Stat2Table
    page::Int
    name::AbstractString
    question:: AbstractString
    feature::Vector{Union{AbstractString,Symbol}}
end


"""
    plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String,save::Bool=false)
    使用两个 feature 绘制散点图
    ## Params
    1. data::DataFrame
    2. xlabel::  x feature
    3. ylabel::  y feature
    4. save:: 是否保存图片 默认 false
    ## 返回值
       fig,ax
"""
function plot_pair_scatter(data::AbstractDataFrame;xlabel::String,ylabel::String)
    fig=Figure()
    ax=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax.title="$(xlabel)-$(ylabel)-scatter"
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.5)
    scatter!(ax,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    #res= save==true ? save("$(xlabel)-$(ylabel)-scatter.png",fig) :  
    return (fig,ax)
    
end


"""
    plot_fitline_and_residual(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    绘制回归模型图: fig[1,1] 散点+拟合线, fig[1,2] 预测残差图

 ## Params
    1. data   df
    2. xlabel 预测变量  
    3. ylabel  响应变量
    4. model   回归模型
 ##  返回值
    fig  Makie 对象
"""
function plot_fitline_and_residual(;data::AbstractDataFrame,xlabel::Union{String,Symbol}
    ,ylabel::Union{String,Symbol},model::RegressionModel)
    y_hat=@pipe select(df,xlabel)|>predict(model,_)|>round.(_,digits=2)
    res=residuals(model)
    fig=Figure(resolution=(800,300))
    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    #ax1.title="$(xlabel)-$(ylabel)-scatter"
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.5)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.5)
    scatter!(ax1,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    lines!(ax1,data[!,xlabel],y_hat,label="fit_line")
    stem!(ax2,res)
    return fig
end

"""
    plot_lm_res(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    绘制回归模型图: fig[1,1] 散点+拟合线, fig[1,2] 预测残差图,fig[2,1] residuals histrogram,fig[2,2]  residuals qqnorm
 
 ## Params
    1. data   df
    2. xlabel 预测变量  
    3. ylabel  响应变量
    4. model   回归模型
 ##  返回值
    fig  Makie 对象
"""
function plot_lm_res(;data::AbstractDataFrame,xlabel::Union{String,Symbol}
    ,ylabel::Union{String,Symbol},model::RegressionModel)
    y_hat=@pipe select(data,xlabel)|>predict(model,_)|>round.(_,digits=2)
    res=residuals(model)
    fig=Figure(resolution=(1000,800))
    Label(fig[0, 1:2], "$(xlabel)-$(ylabel)-Linear-Regression", fontsize = 24)

    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    ax3=Axis(fig[2,1],xlabel="rediduals",ylabel="frequency")
    ax4=Axis(fig[2,2],xlabel="quantiles",ylabel="residuals")
    
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,2];color = (:orange,0.1),strokewidth=0.3)
    scatter!(ax1,data[!,xlabel],data[!,ylabel];marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    lines!(ax1,data[!,xlabel],y_hat,label="fit_line")
    stem!(ax2,res)
    hist!(ax3,res)
    qqnorm!(ax4,res;qqline = :fitrobust)
    return fig
end


"""
    plot_lm_res2(;data::AbstractDataFrame,xlabel::Union{String,Symbol},ylabel::Union{String,Symbol},model::RegressionModel)
    与 plot_lm_res 功能相同, y_hat 改为平方
    
`
xs=sort(data[!,xlabel])

y_hat=@pipe select(df,xlabel)|>predict(model,_)|>_.^2|>round.(_,digits=2)|>sort
`TBW
"""
function plot_lm_res2(;data::AbstractDataFrame,xlabel::Union{String,Symbol}
    ,ylabel::Union{String,Symbol},model::RegressionModel)
    xs=sort(data[!,xlabel])
    y_hat=@pipe select(data,xlabel)|>predict(model,_)|>_.^2|>round.(_,digits=2)|>sort
    
    res=residuals(model)
    
    fig=Figure(resolution=(1000,800))
    Label(fig[0, 1:2], "$(xlabel)-$(ylabel)-Linear-Regression", fontsize = 24)

    ax1=Axis(fig[1,1],xlabel=xlabel,ylabel=ylabel)
    ax2=Axis(fig[1,2],xlabel="fit_value",ylabel="residuals")
    ax3=Axis(fig[2,1],xlabel="rediduals",ylabel="frequency")
    ax4=Axis(fig[2,2],xlabel="quantiles",ylabel="residuals")
    
    Box(fig[1,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[1,2];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,1];color = (:orange,0.1),strokewidth=0.3)
    Box(fig[2,2];color = (:orange,0.1),strokewidth=0.3)
    scatter!(ax1,data[!,xlabel],data[!,ylabel].^2;marker=:circle,markersize=14,color=(:purple,0.4),strokewidth=1,strokecolor=:black)
    scatterlines!(ax1,xs,y_hat,label="fit_line")
    stem!(ax2,res)
    hist!(ax3,res)
    qqnorm!(ax4,res;qqline = :fitrobust)
    return fig
end


"""
    plot_cor_group(data::Union{SubDataFrame,AbstractDataFrame})
    plot 多组配对属性scatter
    `预测变量放在 dataframe feature 的最后一位,便于绘图`

"""
function plot_cor_group(data::Union{SubDataFrame,AbstractDataFrame})
        cats=names(data) #获取列名(属性) 
        combinations_x = combinations(1:length(cats),2) #使用combinations 组合生成配对属性
        row,_=fldmod1(length(combinations_x),3)
        cbarPal = :thermal
        cmap = cgrad(colorschemes[cbarPal], length(combinations_x), categorical = true)
        fig=Figure(resolution=(1200,row*300))
        
        for (idx,c) in enumerate(combinations_x)
            local row,col=fldmod1(idx,3) # 生成三列的 layout
            local ax = Axis(fig[row, col],xlabel=cats[c[1]],ylabel=cats[c[2]])
            Box(fig[row, col], color = (:orange,0.1))
            scatter!(ax,data[!,c[1]],data[!,c[2]];marker=:circle,markersize=12,color=(cmap[idx],0.2),strokewidth=2,strokecolor=:black)
        end
        fig
        #save("./imgs/iris-scatter-plot.png",fig)
    end


"""
   
    plot_residuals_qq(model::RegressionModel)
    绘制 残差和 qq 图

## Arguments
       model::RegressionModel

## 返回  GLMakie  fig 对象

"""
function plot_residuals_qq(model::RegressionModel)
        res=residuals(model)
        fig=Figure(resolution=(800,400))
        ax1=Axis(fig[1,1],xlabel="Fitted Values",ylabel="Residuals")
        ax2=Axis(fig[1,2],xlabel="Quantiles",ylabel="Residuals")
        Box(fig[1,1];color = (:orange,0.1),strokewidth=0.1)
        Box(fig[1,2];color = (:orange,0.1),strokewidth=0.1)
        stem!(ax1,res)
        qqnorm!(ax2,res;qqline = :fitrobust)
        return fig
end



"""
    plot_group_scatter(gdf::GroupedDataFrame,cats,colors)
    同一张上绘制多组 scatter 点
## Arguments
   gdf:   分组 dataframe
   cats:  分组类名
   colors: 颜色数组
"""
function plot_group_scatter(gdf::GroupedDataFrame,cats,colors)
    
    fig=Figure(resolution=(800,600))
    ax=Axis(fig[1,1])
    for (idx,df) in enumerate(gdf)
        scatter!(ax,df[!,:Age],df[!,:Weight]
        ;marker=:circle,markersize=16,color=(colors[idx],0.5),strokewidth=1,strokecolor=:black,
        label=cats[idx]
        )
    end
    axislegend(ax)
    fig
    
end


"""
    plot_group_scatter2(gdf::GroupedDataFrame,feature,cats,colors)
    feature::Vector{Union{AbstractString,Symbol}},
    cats::Vector{Union{AbstractString,Symbol}},colors::Vector{Symbol})
    分组数据在同一张图上绘制 scatter

"""
function plot_group_scatter2(gdf::GroupedDataFrame,feature,cats,colors)
    
    fig=Figure(resolution=(800,600))
    ax=Axis(fig[1,1];xlabel=feature[1],ylabel=feature[2])
    for (idx,df) in enumerate(gdf)
        scatter!(ax,df[!,feature[1]],df[!,feature[2]]
        ;marker=:circle,markersize=16,color=(colors[idx],0.5),strokewidth=1,strokecolor=:black,
        label=cats[idx]
        )
    end
    axislegend(ax)
    fig
    
end


"""
        computing_vif(data::AbstractDataFrame,formula::FormulaTerm)
        使用R car library 计算 VIF
    ## Arguments
        1. data::  DataFrame
        2. formula 回归公式,GLM.jl 格式
    ## 返回值:  vif  数组

"""
    function computing_vif(data::AbstractDataFrame,formula::FormulaTerm)
        @rput data; @rput formula
        R"""
            library(car)
            model <- lm(formula, data = data)
            vif_data=vif(model)
        """
        return @rget  vif_data
    end


 """
        print_cormatrix(data::AbstractDataFrame,feature::Vector{Union{AbstractString,Symbol}})
        打印 dataframe 的相关性矩阵列表
## Arguments
    1. data 数据 Dataframe, 应该把 ID等 feature 去掉
    2. feature 需要展示的 feature
## Return
    无返回值

"""
function print_cormatrix(data::AbstractDataFrame,feature::Vector{Union{AbstractString,Symbol}})
    Names=DataFrame(Name=String.(feature))
    df=@pipe  data|>Matrix|>cor|>round.(_,digits=2)|>DataFrame(_,feature)
    cormatrix=hcat(Names,df)
    pretty_table(cormatrix)
end


"""
    plot_cormatrix_heatmap(data::AbstractDataFrame,title::String)
    绘制相关矩阵 heatmap
## Arguments
   1. data::DataFrame
   2. name of plot
## 返回值
   Makie  fig 对象
"""
function plot_cormatrix_heatmap(data::AbstractDataFrame,title::String)
    df_cov=@pipe  data|>Matrix|>cor|>round.(_,digits=2)
    label=names(data)
    ran=1:length(label)
    fig = Figure(resolution=(600, 600))
    ax = Axis(fig[1, 1]; xticks=(ran, label), yticks=(ran, label), title="$title cov matrix")
    hm = heatmap!(ax, df_cov;yflip=true)
    Colorbar(fig[1, 2], hm)
    [text!(ax, x, y; text=string(df_cov[x, y]), color=:white, fontsize=18, align=(:center, :center)) for x in ran, y in ran]
    fig
end

#label::Vector{Union{AbstractString,Symbol}}

get_df_feature(df::AbstractDataFrame)=show(names(data))

"""
Makes the walker walk for T timesteps
   
ref: [random-walk](https://kb.katnoria.com/posts/2020/8/random-walk/)
# Arguments
- `w::Walker1D`: one-dimensional walker
- `T:Int64`: Number of timesteps in the walk
"""
function walk(w, T)
    trajectory = [w]
    for i ∈ 1:T
        w = move(w)
        push!(trajectory, deepcopy(w))
    end
    return trajectory
end

marker_style=(marker=:circle,markersize=12,color=(:green,0.2),strokewidth=1,strokecolor=:black)

