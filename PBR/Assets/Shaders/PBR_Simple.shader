Shader "PBR_Simple"{
    Properties{
        _Color("Color", Color) = (1,1,1,1)                       // 颜色
        _MainTex("Albedo (RGB)", 2D) = "white" {}                // 反照率
        _MetallicGlossMap("Metallic", 2D) = "white"{}            // 金属贴图（r金属度，a光滑度）
        _BumpMap("Normal Map", 2D) = "bump" {}                   // 法线贴图
        _OcclusionMap("Occlusion", 2D) = "white"{}               // 环境光遮蔽纹理
        _MetallicStrength("Metallic Strength", Range(0,1)) = 1   // 金属强度
        _GlossStrength("Gloss Strength", Range(0,1)) = 0.5       // 光滑强度
        _BumpScale("Bump Scale", Float) = 1                      // 法线影响
        _EmissionColor("Emission Color", Color) = (0,0,0,0)      // 自发光颜色
        _EmissionMap("Emission Map", 2D) = "white"{}             // 自发光贴图
    }

    CGINCLUDE

        #include "UnityCG.cginc"
        #include "Lighting.cginc"
        #include "AutoLight.cginc"

    // BRDF
        // G:几何遮蔽函数
        // 参数：法线n与光线方向l余弦；法线n与视角方向v余弦；材质粗糙度
        inline half SchlickGGX_Unity5(half nl, half nv, half roughness){
            half k = roughness * roughness / 2;
            half lambdaV = (nv * (1 - k) + k);
            half lambdaL = (nl * (1 - k) + k);
            return 1.0f / (lambdaV * lambdaL);
        }
        inline half SchlickGGX_UE4(half nl, half nv, half roughness){
            half k = (roughness + 1) * (roughness + 1) / 8;
            // half k = (roughness* roughness) / 2;
            half G_v = nv / (nv * (1 - k) + k);
            half G_l = nl / (nl * (1 - k) + k);

            return G_v * G_l;
        }

        // D：法线分布函数
        // 参数：法线n与半程向量h的余弦；粗糙度
        inline half GGXTerm(half nh, half roughness){
            half a = roughness * roughness;
            half a2 = a * a;
            half d = (nh * nh) * (a2 - 1.0f) + 1.0f;
            return a2 * UNITY_INV_PI / (d * d + 1e-5f);
        }

        // F：菲涅尔
		// 参数：基础反射率；法线v与半程向量h的余弦
        inline half3 SchlickFresnel(half3 F0, half hv){
            return F0 + (1 - F0) * pow(1 - hv, 5);
        }

        // burley 漫反射模型
        // 参数：法线n与视角方向v的余弦；法线n与光线方向l的余弦；光线方向l与半程向量h的余弦；粗糙度；基础反射率
        inline half3 BurleyDiffuseTerm(half nv, half nl, half lh, half roughness, half3 baseColor){
            half Fd90 = 0.5f + 2 * roughness * lh * lh;
            return baseColor * UNITY_INV_PI * (1 + (Fd90 - 1) * pow(1 - nl, 5)) * (1 + (Fd90 - 1) * pow(1 - nv, 5));
        }

        // lambert 漫反射模型
        inline half3 LambertDiffuseTerm(half3 baseColor){
            return baseColor * UNITY_INV_PI;
        }

    ENDCG
    SubShader{
        Tags{"RenderType" = "Opaque"}
        pass{
            Tags{"LightMode" = "ForwardBase"}
            CGPROGRAM
            #pragma target 3.0

            #pragma multi_compile_fwdbase
            #pragma multi_compile_fog

            #pragma vertex vert
            #pragma fragment frag

            half4 _Color;
			sampler2D _MainTex;
			float4 _MainTex_ST;
			sampler2D _MetallicGlossMap;
			sampler2D _BumpMap;
			sampler2D _OcclusionMap;
			half _MetallicStrength;
			half _GlossStrength;
			float _BumpScale;
			half4 _EmissionColor;
			sampler2D _EmissionMap;

            struct a2v
			{
				float4 vertex : POSITION;
				float3 normal : NORMAL;
				float4 tangent :TANGENT;
				float2 texcoord : TEXCOORD0;
				float2 texcoord1 : TEXCOORD1;
				float2 texcoord2 : TEXCOORD2;
			};
            
            struct v2f
			{
				float4 pos : SV_POSITION;
				float2 uv : TEXCOORD0;
				float4 TtoW0 : TEXCOORD1;
				float4 TtoW1 : TEXCOORD2;
				float4 TtoW2 : TEXCOORD3;//xyz 存储着 从切线空间到世界空间的矩阵，w存储着世界坐标
                half4 ambientOrLightmapUV : TEXCOORD4;//存储环境光或光照贴图的UV坐标
				SHADOW_COORDS(5) //定义阴影所需要的变量(AutoLight.cginc)
				UNITY_FOG_COORDS(6) //定义雾效所需要的变量(UnityCG.cginc)
			};

            v2f vert(a2v v){
                v2f o;
                UNITY_INITIALIZE_OUTPUT(v2f, o); //初始化结构体数据，定义在HLSLSupport.cginc

                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.texcoord, _MainTex);

                float3 worldPos = mul(unity_ObjectToWorld, v.vertex);
                half3 worldNormal = UnityObjectToWorldNormal(v.normal);
                half3 worldTangent = UnityObjectToWorldDir(v.tangent);
                half3 worldBinormal = cross(worldNormal, worldTangent) * v.tangent.w;

                o.TtoW0 = float4(worldTangent.x, worldBinormal.x, worldNormal.x, worldPos.x);
                o.TtoW1 = float4(worldTangent.y, worldBinormal.y, worldNormal.y, worldPos.y);
                o.TtoW2 = float4(worldTangent.z, worldBinormal.z, worldNormal.z, worldPos.z);

                //填充阴影所需要的参数(AutoLight.cginc)
				TRANSFER_SHADOW(o);
				//填充雾效所需要的参数(UnityCG.cginc)
				UNITY_TRANSFER_FOG(o,o.pos);

                return o;
            }

            half4 frag(v2f i) : SV_Target{
                // 数据准备
                float3 worldPos = float3(i.TtoW0.w, i.TtoW1.w, i.TtoW2.w);  
                half3 albedo = tex2D(_MainTex, i.uv).rgb * _Color.rgb;      // 反照率（漫反射）
                half2 metallicGloss = tex2D(_MetallicGlossMap, i.uv).ra;         
                half metallic = metallicGloss.x * _MetallicStrength;         // 金属度
                half roughness = 1 - metallicGloss.y * _GlossStrength;       // 粗糙度
                half occlusion = tex2D(_OcclusionMap, i.uv).g;               // 环境光遮挡

                // 法线（世界空间）
                half3 normalTangent = UnpackNormal(tex2D(_BumpMap, i.uv));
                normalTangent.xy *= _BumpScale;
                normalTangent.z = sqrt(1.0 - saturate(dot(normalTangent.xy, normalTangent.xy)));
                half3 worldNormal = normalize(half3(dot(i.TtoW0.xyz, normalTangent),
                                                    dot(i.TtoW1.xyz, normalTangent),
                                                    dot(i.TtoW2.xyz, normalTangent)));

                half3 lightDir = normalize(UnityWorldSpaceLightDir(worldPos));  // 光线方向WS(UnityCG.cginc)
                half3 viewDir = normalize(UnityWorldSpaceViewDir(worldPos));    // 视线方向WS(UnityCG.cginc)
                half3 refDir = reflect(-viewDir, worldNormal);                  // 反射方向WS

                // 自发光
                half3 emission = tex2D(_EmissionMap, i.uv).rgb * _EmissionColor;    

                UNITY_LIGHT_ATTENUATION(atten, i, worldPos);                //计算阴影和衰减(AutoLight.cginc)

                // 计算BRDF需要的项
                half3 halfDir = normalize(lightDir + viewDir);
                half nv = saturate(dot(worldNormal, viewDir));
                half nl = saturate(dot(worldNormal, lightDir));
                half nh = saturate(dot(worldNormal, halfDir));
                half lh = saturate(dot(lightDir, halfDir));

                // 镜面反射率
                // - 纯金属只有镜面反射，所以金属度代表了镜面反射程度
                // - unity_ColorSpaceDielectricSpec.a 非金属材质的反射率（默认是 0.04）
                half3 specColor = lerp(unity_ColorSpaceDielectricSpec.rgb, albedo, metallic);
                half oneMinusReflectivity = (1 - metallic) * unity_ColorSpaceDielectricSpec.a;
                // 漫反射率
                half3 diffColor = albedo * oneMinusReflectivity;

            // 【BRDF】直射光

                // BRDF 高光反射项
                half G = SchlickGGX_UE4 (nl, nv, roughness);              // G几何遮蔽函数：（Unity）
                // half G = SchlickGGX_Unity5(nl,nv,roughness);           
                half D = GGXTerm(nh, roughness);                          // D法线分布函数
                half3 F = SchlickFresnel(specColor, lh);                     // F菲涅尔
                half3 specularTerm = G * D * F / (4.0 * max(nv * nl, 0.001)); 

                // BRDF 漫反射项
                // half3 diffuseTerm = BurleyDiffuseTerm(nv, nl, lh, roughness, diffColor);
                half3 diffuseTerm = LambertDiffuseTerm(diffColor);

                // 【反射方程】
                half3 L_o = UNITY_PI * (diffuseTerm + specularTerm) * _LightColor0.rgb * nl;

                // 环境光
                float3 ambient = UNITY_LIGHTMODEL_AMBIENT.rgb * albedo;

                // 最终颜色
                half3 color = L_o * atten + ambient+ emission;
               
                //雾效(UnityCG.cginc)
				UNITY_APPLY_FOG(i.fogCoord, color.rgb);

                return half4(color,1);
            }
            ENDCG
        }
    }
    FallBack "VertexLit"
}